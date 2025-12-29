import os
import torch
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from medical_config import DEVICE, MODEL_PATH, DISEASE_SYMPTOM_TYPES

app = Flask(__name__)

tokenizer = None
model = None

def get_recommended_department(prediction):
    m = {
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
    return m.get(prediction, '全科')

def load_model_and_tokenizer():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    src = './best_disease_model' if os.path.isdir('./best_disease_model') else MODEL_PATH
    try:
        tokenizer = BertTokenizer.from_pretrained(src)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    try:
        model = BertForSequenceClassification.from_pretrained(
            src,
            num_labels=len(DISEASE_SYMPTOM_TYPES),
            output_attentions=False,
            output_hidden_states=False,
            use_safetensors=False
        )
    except Exception:
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=len(DISEASE_SYMPTOM_TYPES),
            output_attentions=False,
            output_hidden_states=False
        )
    model = model.to(DEVICE)

def diagnose(text):
    load_model_and_tokenizer()
    model.eval()
    enc = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(torch.argmax(logits, dim=1).cpu().item())
    return {
        'text': text,
        'type_name': DISEASE_SYMPTOM_TYPES.get(pred, '未知疾病类型'),
        'type_id': pred,
        'prob': float(probs[pred]),
        'probs': [float(p) for p in probs.tolist()],
        'department': get_recommended_department(pred)
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None, types=DISEASE_SYMPTOM_TYPES)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '').strip()
    result = diagnose(symptoms) if symptoms else None
    return render_template('index.html', result=result, types=DISEASE_SYMPTOM_TYPES)

if __name__ == '__main__':
    load_model_and_tokenizer()
    app.run(host='0.0.0.0', port=5000, debug=False)