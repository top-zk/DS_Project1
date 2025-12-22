import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from medical_config import DEVICE, MAX_LEN, BATCH_SIZE, EPOCHS, MODEL_PATH
from medical_config import DISEASE_SYMPTOM_TYPES, DISEASE_SYMPTOM_DESCRIPTIONS, DIAGNOSTIC_CRITERIA
from medical_config import TRAIN_DATA_PATH, VAL_DATA_PATH
from medical_data_loader import load_disease_data_from_txt


class DiseaseDataset(Dataset):
    """疾病症状数据集"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_disease_model(train_loader, val_loader, model, device, epochs=3, lr=2e-5):
    """训练疾病诊断模型"""
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_accuracy = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        val_accuracy = 0
        val_loss = 0
        if val_loader:
            model.eval()
            total_val_accuracy = 0
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    total_val_loss += loss.item()

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == labels).cpu().numpy().mean()
                    total_val_accuracy += accuracy

            val_accuracy = total_val_accuracy / len(val_loader)
            val_loss = total_val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(
            f'Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f}' if val_loader else f'Train loss: {avg_train_loss:.4f}')
        if val_loader:
            print(f'Val accuracy: {val_accuracy:.4f}')

        # 保存最佳模型
        if val_loader and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_disease_model.bin')
            print("保存最佳疾病诊断模型")

    if val_loader:
        print(f'疾病诊断模型训练完成! 最佳验证准确率: {best_accuracy:.4f}')
    else:
        print('疾病诊断模型训练完成!')


def diagnose_disease_symptoms(text, model, tokenizer, device, max_len=128):
    """诊断疾病症状类型"""
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    prediction = torch.argmax(logits, dim=1).cpu().item()

    return {
        '症状描述': text,
        '疾病类型': DISEASE_SYMPTOM_TYPES.get(prediction, '未知疾病类型'),
        '类型编码': prediction,
        '症状特征': DISEASE_SYMPTOM_DESCRIPTIONS.get(prediction, '未知症状特征'),
        '诊断概率': probabilities[prediction],
        '概率分布': probabilities,
        '推荐科室': get_recommended_department(prediction),
        '诊断标准': DIAGNOSTIC_CRITERIA.get(prediction, '请结合临床检查确认')
    }


def get_recommended_department(prediction):
    """根据疾病类型推荐就诊科室"""
    department_mapping = {
        0: '呼吸内科',
        1: '心血管内科',
        2: '消化内科',
        3: '神经内科',
        4: '骨科/康复科',
        5: '皮肤科',
        6: '泌尿外科/肾内科',
        7: '全科/内科',
        8: '耳鼻喉科/眼科',
        9: '精神心理科'
    }
    return department_mapping.get(prediction, '全科')


def main():
    # 加载疾病数据
    train_data = load_disease_data_from_txt(TRAIN_DATA_PATH)
    val_data = load_disease_data_from_txt(VAL_DATA_PATH) if os.path.exists(VAL_DATA_PATH) else []

    train_texts = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]

    val_texts = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]

    print(f"训练集: {len(train_texts)}条疾病症状数据, 验证集: {len(val_texts)}条疾病症状数据")

    # 初始化分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        print("分词器加载成功")
    except Exception as e:
        print(f"分词器加载失败: {e}")
        print("使用默认分词器")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 初始化疾病诊断模型
    try:
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(DISEASE_SYMPTOM_TYPES),
            output_attentions=False,
            output_hidden_states=False,
            use_safetensors=False
        )
        model = model.to(DEVICE)
        print("疾病诊断模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试从在线源下载模型...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese",
            num_labels=len(DISEASE_SYMPTOM_TYPES),
            output_attentions=False,
            output_hidden_states=False
        )
        model = model.to(DEVICE)
        print("在线疾病诊断模型加载成功")

    # 创建疾病数据集和数据加载器
    train_dataset = DiseaseDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = None
    if val_texts:
        val_dataset = DiseaseDataset(val_texts, val_labels, tokenizer, MAX_LEN)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 训练疾病诊断模型
    train_disease_model(train_loader, val_loader, model, DEVICE, EPOCHS)

    # 保存疾病诊断模型
    model.save_pretrained('./disease_model')
    tokenizer.save_pretrained('./disease_model')
    print("疾病诊断模型保存完成")

    # 测试诊断功能
    test_symptoms = [
        "咳嗽发热呼吸困难胸痛",
        "胸闷心悸气短胸痛",
        "腹痛腹泻恶心呕吐",
        "头痛头晕失眠记忆力减退",
        "关节疼痛腰背痛活动受限"
    ]

    print("\n=== 疾病症状诊断测试 ===")
    for symptom in test_symptoms:
        diagnosis = diagnose_disease_symptoms(symptom, model, tokenizer, DEVICE)
        print(f"症状描述: {diagnosis['症状描述']}")
        print(f"疾病类型: {diagnosis['疾病类型']}")
        print(f"症状特征: {diagnosis['症状特征']}")
        print(f"推荐科室: {diagnosis['推荐科室']}")
        print(f"诊断概率: {diagnosis['诊断概率']:.4f}")
        print(f"诊断标准: {diagnosis['诊断标准']}")
        print("-" * 60)


if __name__ == "__main__":
    main()