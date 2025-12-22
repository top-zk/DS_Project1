# train.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from config import DEVICE, MAX_LEN, BATCH_SIZE, EPOCHS, MODEL_PATH, INTENT_TYPES
from config import TRAIN_DATA_PATH, VAL_DATA_PATH
from data_loader import load_data_from_txt


class QuestionDataset(Dataset):
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


def train_model(train_loader, val_loader, model, device, epochs=3, lr=2e-5):
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
            torch.save(model.state_dict(), 'best_model.bin')
            print("保存最佳模型")

    if val_loader:
        print(f'训练完成! 最佳验证准确率: {best_accuracy:.4f}')
    else:
        print('训练完成!')


def predict_intent(text, model, tokenizer, device, max_len=128):
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
        'text': text,
        'intent': INTENT_TYPES.get(prediction, '未知意图'),
        'prediction': prediction,
        'probabilities': probabilities
    }


def main():
    # 加载数据
    train_data = load_data_from_txt(TRAIN_DATA_PATH)
    val_data = load_data_from_txt(VAL_DATA_PATH) if os.path.exists(VAL_DATA_PATH) else []

    train_texts = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]

    val_texts = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]

    print(f"训练集: {len(train_texts)}条, 验证集: {len(val_texts)}条")

    # 初始化分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        print("分词器加载成功")
    except Exception as e:
        print(f"分词器加载失败: {e}")
        print("使用默认分词器")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 初始化模型
    try:
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=len(INTENT_TYPES),
            output_attentions=False,
            output_hidden_states=False,
            use_safetensors=False
        )
        model = model.to(DEVICE)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试从在线源下载模型...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese",
            num_labels=len(INTENT_TYPES),
            output_attentions=False,
            output_hidden_states=False
        )
        model = model.to(DEVICE)
        print("在线模型加载成功")

    # 创建数据集和数据加载器
    train_dataset = QuestionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = None
    if val_texts:
        val_dataset = QuestionDataset(val_texts, val_labels, tokenizer, MAX_LEN)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 训练模型
    train_model(train_loader, val_loader, model, DEVICE, EPOCHS)

    # 保存模型
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')
    print("模型保存完成")



if __name__ == "__main__":
    main()