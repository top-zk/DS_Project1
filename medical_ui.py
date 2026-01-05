import os
import torch
from flask import Flask, render_template, request, session, redirect, url_for, flash
from transformers import BertTokenizer, BertForSequenceClassification
from medical_config import DEVICE, MODEL_PATH, DISEASE_SYMPTOM_TYPES, DISEASE_SYMPTOM_KEYWORDS
from models import db, init_db, add_user, verify_user, get_encyclopedia_letters, get_articles_by_letter, search_articles, get_article_by_id

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_medical_ai_app'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

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
    
    # ---------------------------------------------------------
    # Rule-based Adjustment (Keyword Matching)
    # ---------------------------------------------------------
    # Boost logits for categories where keywords appear in text
    rule_boost = torch.zeros_like(logits)
    found_keywords = False
    
    for type_id, keywords in DISEASE_SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                # Add significant boost if keyword found
                # The boost value (e.g., 5.0) should be tuned. 
                # Since BERT logits can vary, adding 5.0 is usually enough to sway the softmax significantly.
                rule_boost[0, type_id] += 3.0 
                found_keywords = True
                
    if found_keywords:
        logits = logits + rule_boost
        
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
    return render_template('home.html')

@app.route('/diagnose', methods=['GET'])
def diagnose_page():
    return render_template('diagnose.html', result=None, types=DISEASE_SYMPTOM_TYPES)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = verify_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['first_name'] or user['email']
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        else:
            flash('邮箱或密码错误', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    print(f"Register route accessed. Method: {request.method}")
    if request.method == 'POST':
        print("Processing registration form data...")
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password_confirm')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        print(f"Registering user: {email}, {first_name} {last_name}")

        if password != password_confirm:
            print("Password mismatch")
            flash('两次输入的密码不一致', 'danger')
            return render_template('register.html')
        
        try:
            if add_user(email, password, first_name, last_name):
                print("User added successfully")
                flash('注册成功，请登录', 'success')
                return redirect(url_for('login'))
            else:
                print("User add failed (email likely exists)")
                flash('注册失败，该邮箱可能已被注册', 'danger')
        except Exception as e:
            print(f"Exception during add_user: {e}")
            flash(f'注册发生错误: {str(e)}', 'danger')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('您已退出登录', 'info')
    return redirect(url_for('index'))

@app.route('/encyclopedia')
def encyclopedia():
    letter = request.args.get('letter', 'A')
    query = request.args.get('q')
    
    letters = get_encyclopedia_letters()
    
    if query:
        articles = search_articles(query)
        current_letter = None
    else:
        # If letters list is not empty and letter is not in it (and not default A), fallback?
        # For now just trust the query or DB.
        articles = get_articles_by_letter(letter)
        current_letter = letter
        
    return render_template('encyclopedia.html', 
                           letters=letters, 
                           articles=articles, 
                           current_letter=current_letter,
                           query=query)

@app.route('/encyclopedia/<int:article_id>')
def encyclopedia_detail(article_id):
    article = get_article_by_id(article_id)
    if not article:
        flash('未找到相关文章', 'warning')
        return redirect(url_for('encyclopedia'))
    return render_template('encyclopedia_detail.html', article=article)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '').strip()
    
    result = diagnose(symptoms) if symptoms else None
    return render_template('diagnose.html', result=result, types=DISEASE_SYMPTOM_TYPES)

if __name__ == '__main__':
    try:
        with app.app_context():
            init_db(app)
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    
    load_model_and_tokenizer()
    print("Model loaded. Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
