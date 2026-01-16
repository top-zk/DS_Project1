import os
import torch
from flask import Flask, render_template, request, session, redirect, url_for, flash
from transformers import BertTokenizer, BertForSequenceClassification
from medical_config import DEVICE, MODEL_PATH, DISEASE_SYMPTOM_TYPES, DISEASE_SYMPTOM_KEYWORDS, DISEASE_SYMPTOM_DESCRIPTIONS, DIAGNOSTIC_CRITERIA
from models import db, init_db, add_user, verify_user, get_encyclopedia_letters, get_articles_by_letter, search_articles, get_article_by_id, log_interaction, get_related_diseases, SymptomDiseaseRule
from sqlalchemy import func

FOLLOW_UP_QUESTIONS = {
    0: ["请问您是否发烧？体温大概是多少？", "是否有咳嗽或咳痰的情况？痰的颜色是什么？", "是否感觉呼吸困难或气促？"],
    1: ["请问您是否有胸痛或胸闷的感觉？", "这种感觉是持续性的还是阵发性的？", "是否有心悸或心跳加速的感觉？"],
    2: ["请问具体的腹痛位置在哪里？", "是否有恶心、呕吐或腹泻的症状？", "最近的饮食情况如何？"],
    3: ["请问头痛的具体部位是哪里？", "是否有头晕或眩晕的感觉？", "睡眠质量如何？是否失眠？"],
    4: ["请问具体是哪个关节或部位疼痛？", "疼痛是否影响活动？", "是否有受过外伤？"],
    5: ["请问皮疹或瘙痒出现的部位在哪里？", "是否有红肿或脱皮？", "症状出现多久了？"],
    6: ["请问是否有尿频、尿急或尿痛的症状？", "尿液颜色是否正常？", "是否有腰痛？"],
    7: ["请问发热持续了多久？", "是否伴有乏力或体重下降？", "是否有盗汗现象？"],
    8: ["请问具体是眼睛、耳朵、鼻子还是喉咙不舒服？", "是否有视力下降或听力下降？", "是否有流血或异常分泌物？"],
    9: ["请问这种情绪持续了多久？", "是否影响到了日常生活或工作？", "是否有睡眠障碍？"]
}

def extract_keywords_from_rules(text):
    """
    Scan text against all keywords present in the SymptomDiseaseRule table.
    Returns a list of unique matched keyword strings.
    """
    matched = set()
    try:
        # Get all rules
        rules = SymptomDiseaseRule.query.all()
        # Flatten all keyword combinations into a single set of unique symptoms
        all_symptoms = set()
        for r in rules:
            if r.keywords:
                for k in r.keywords.split(','):
                    all_symptoms.add(k.strip())
        
        # Check if any symptom is in the text
        for s in all_symptoms:
            if s and s in text:
                matched.add(s)
                
    except Exception as e:
        print(f"Error extracting keywords: {e}")
    return list(matched)

def get_next_question(current_keywords):
    """
    Find the best matching rule and generate a question about missing symptoms.
    """
    if not current_keywords:
        return None, None

    try:
        # Find rules that overlap with current_keywords
        # We score rules by intersection size
        rules = SymptomDiseaseRule.query.all()
        best_rule = None
        best_overlap = 0
        
        current_set = set(current_keywords)
        
        for rule in rules:
            rule_kws = set(rule.keywords.split(',')) if rule.keywords else set()
            overlap = len(current_set.intersection(rule_kws))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_rule = rule
            elif overlap == best_overlap and best_rule:
                pass
        
        if best_rule and best_overlap > 0:
            # Found a candidate rule.
            # Find missing keywords from this rule
            rule_kws = set(best_rule.keywords.split(',')) if best_rule.keywords else set()
            missing = list(rule_kws - current_set)
            
            if missing:
                # Ask about missing symptoms
                # Pick up to 3 to ask
                to_ask = missing[:3]
                question = f"为了更准确地判断是否为{best_rule.disease_name}，请问您是否还有以下症状：{', '.join(to_ask)}？"
                return question, best_rule
                
    except Exception as e:
        print(f"Error generating question: {e}")
    
    return None, None

def match_suspected_disease(all_keywords, predicted_department=None):
    """
    Find the most likely disease based on collected keywords.
    """
    best_match = None
    
    try:
        if all_keywords:
            # Score all rules based on overlap count
            rules = SymptomDiseaseRule.query.all()
            best_rule = None
            max_score = 0
            
            user_kws = set(all_keywords)
            
            for rule in rules:
                rule_kws = set(rule.keywords.split(',')) if rule.keywords else set()
                overlap = len(user_kws.intersection(rule_kws))
                
                if overlap > max_score:
                    max_score = overlap
                    best_rule = rule
            
            if best_rule:
                return {
                    'name': best_rule.disease_name,
                    'count': max_score,
                    'department': best_rule.department,
                    'advice': best_rule.advice,
                    'source': 'keywords'
                }

        # 2. Fallback: Department Matching
        if not best_match and predicted_department:
            fallback = db.session.query(
                SymptomDiseaseRule.disease_name,
                SymptomDiseaseRule.department,
                SymptomDiseaseRule.advice
            ).filter(
                SymptomDiseaseRule.department == predicted_department
            ).first() # Just pick one
            
            if fallback:
                return {
                    'name': fallback.disease_name,
                    'count': 0,
                    'department': fallback.department,
                    'advice': fallback.advice,
                    'source': 'department_fallback'
                }
                
    except Exception as e:
        print(f"Error matching disease: {e}")
    return None

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_medical_ai_app'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.template_filter('highlight')
def highlight_filter(s, query):
    if not query or not s:
        return s
    import re
    from markupsafe import Markup
    
    keywords = query.strip().split()
    # Sort keywords by length descending to handle overlapping phrases
    keywords.sort(key=len, reverse=True)
    
    for kw in keywords:
        if kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            s = pattern.sub(lambda m: f'<mark>{m.group(0)}</mark>', str(s))
            
    return Markup(s)

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
            output_hidden_states=False
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
    
    # Identify matched keywords for explanation
    matched_keywords = []
    if pred in DISEASE_SYMPTOM_KEYWORDS:
        for kw in DISEASE_SYMPTOM_KEYWORDS[pred]:
            if kw in text:
                matched_keywords.append(kw)
    
    explanation = f"模型根据症状描述识别出 {len(matched_keywords)} 个关键特征 ({', '.join(matched_keywords)})。" if matched_keywords else "模型根据语义分析判断。"
    
    # --- New: Find Specific Disease Match ---
    # Attempt to find the most relevant article in the encyclopedia
    specific_disease_match = None
    try:
        # Search for articles using the input text
        # This uses the same search logic as the encyclopedia page (including translation)
        search_res = search_articles(text, page=1, per_page=1)
        if search_res and search_res.items:
            specific_disease_match = search_res.items[0]
    except Exception as e:
        print(f"Error finding specific disease match: {e}")

    result = {
        'text': text,
        'type_name': DISEASE_SYMPTOM_TYPES.get(pred, '未知疾病类型'),
        'type_id': pred,
        'prob': float(probs[pred]),
        'probs': [float(p) for p in probs.tolist()],
        'department': get_recommended_department(pred),
        'description': DISEASE_SYMPTOM_DESCRIPTIONS.get(pred, ''),
        'criteria': DIAGNOSTIC_CRITERIA.get(DISEASE_SYMPTOM_TYPES.get(pred), ''),
        'explanation': explanation
    }

    if specific_disease_match:
        result['specific_disease'] = {
            'id': specific_disease_match.id,
            'name': specific_disease_match.disease_name,
            'symptoms_snippet': specific_disease_match.symptoms[:100] + '...' if specific_disease_match.symptoms else ''
        }
        
    return result

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
    page = request.args.get('page', 1, type=int)
    
    letters = get_encyclopedia_letters()
    
    if query:
        pagination = search_articles(query, page=page)
        current_letter = None
    else:
        pagination = get_articles_by_letter(letter, page=page)
        current_letter = letter
        
    return render_template('encyclopedia.html', 
                           letters=letters, 
                           pagination=pagination, 
                           current_letter=current_letter,
                           query=query)

@app.route('/encyclopedia/<int:article_id>')
def encyclopedia_detail(article_id):
    article = get_article_by_id(article_id)
    if not article:
        flash('Article not found', 'warning')
        return redirect(url_for('encyclopedia'))
    
    related_articles = get_related_diseases(article_id)
    return render_template('encyclopedia_detail.html', article=article, related_articles=related_articles)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '').strip()
    
    result = None
    if symptoms:
        result = diagnose(symptoms)
        # Log interaction
        user_id = session.get('user_id')
        log_interaction(user_id, symptoms, result)
        
    return render_template('diagnose.html', result=result, types=DISEASE_SYMPTOM_TYPES)

def generate_report(context):
    symptoms = context.get('symptoms_text', '')
    all_keywords = context.get('collected_keywords', [])
    
    # 1. AI Prediction (BERT)
    result = diagnose(symptoms)
    
    # 2. Rule-based Specific Disease Matching
    suspected_case = match_suspected_disease(all_keywords, predicted_department=result['department'])
    
    suspected_html = ""
    if suspected_case:
        trigger_info = f"关键词：{', '.join(all_keywords)}" if suspected_case['source'] == 'keywords' else "基于症状类型推断"
        suspected_html = f"""
        <div class="alert alert-info mt-3">
            <h6 class="alert-heading"><i class="fas fa-search-plus me-2"></i>疑似具体病例分析</h6>
            <p class="mb-1">根据您的描述（{trigger_info}），系统分析疑似为：<strong>{suspected_case['name']}</strong></p>
            <p class="mb-1"><strong>推荐科室：</strong>{suspected_case['department']}</p>
            <p class="mb-0 small text-muted"><strong>建议：</strong>{suspected_case['advice']}</p>
        </div>
        """

    report = f"""
    <div class="card mt-3 border-success">
        <div class="card-header bg-success text-white">
            <i class="fas fa-file-medical-alt me-2"></i>智能推测报告
        </div>
        <div class="card-body">
            <h5 class="card-title text-success">初步诊断：{result['type_name']}</h5>
            <hr>
            {suspected_html}
            <p class="card-text"><strong><i class="fas fa-notes-medical me-2"></i>综合症状描述：</strong><br>{symptoms}</p>
            <p class="card-text"><strong><i class="fas fa-hospital-user me-2"></i>推荐科室：</strong> {result['department']}</p>
            
            <p class="card-text"><strong><i class="fas fa-info-circle me-2"></i>相关说明：</strong><br>{result['description']}</p>
            <p class="card-text"><strong><i class="fas fa-clipboard-check me-2"></i>诊断标准参考：</strong><br>{result['criteria']}</p>
            
            <div class="alert alert-warning mt-3 mb-0">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <small><strong>免责声明：</strong> 本报告由AI生成，仅供参考，不能作为最终医疗诊断依据。请务必前往正规医院就医。</small>
            </div>
        </div>
    </div>
    """
    return report

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = [{
            'role': 'bot',
            'content': '您好！我是您的智能医疗助手。请告诉我您哪里不舒服，或者有什么症状？'
        }]
        session['chat_state'] = 'init'
        session['chat_round'] = 0 # Round counter
        session['chat_context'] = {
            'collected_keywords': [] # Store all keywords found
        }
    
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if user_input:
            # Add user message
            session['chat_history'].append({'role': 'user', 'content': user_input})
            
            # --- Keyword Extraction Step ---
            current_keywords = extract_keywords_from_rules(user_input)
            context = session.get('chat_context', {})
            # Merge new keywords (avoid duplicates)
            existing_kws = set(context.get('collected_keywords', []))
            for k in current_keywords:
                existing_kws.add(k)
            context['collected_keywords'] = list(existing_kws)
            
            # Increment round
            current_round = session.get('chat_round', 0) + 1
            session['chat_round'] = current_round
            
            session.modified = True
            
            # Process logic
            state = session.get('chat_state', 'init')
            
            bot_reply = "抱歉，我没有理解，请再说一遍。"
            
            # Check if we reached round limit (5 rounds)
            if current_round >= 5:
                session['chat_state'] = 'report'
                # Update context with full text for BERT diagnose
                context['symptoms_text'] = context.get('symptoms_text', '') + f"；{user_input}"
                session['chat_context'] = context
                bot_reply = generate_report(context)
            else:
                if state == 'init':
                    # First diagnosis
                    result = diagnose(user_input)
                    prob = result['prob']
                    pred_type = result['type_id']
                    
                    # Try to find a dynamic question based on rules first
                    rule_question, _ = get_next_question(context.get('collected_keywords', []))
                    
                    if rule_question:
                        session['chat_state'] = 'asking_details'
                        context['type_id'] = pred_type
                        context['symptoms_text'] = user_input
                        # Use special flag to indicate we are using dynamic rule questions
                        context['use_dynamic_questions'] = True 
                        session['chat_context'] = context
                        bot_reply = f"根据您的描述，初步推测可能与{result['type_name']}有关。{rule_question}"
                    
                    elif prob > 0.4: 
                        session['chat_state'] = 'asking_details'
                        context['type_id'] = pred_type
                        context['symptoms_text'] = user_input
                        context['question_idx'] = 0
                        context['use_dynamic_questions'] = False
                        session['chat_context'] = context
                        
                        # Ask first question from legacy list
                        questions = FOLLOW_UP_QUESTIONS.get(pred_type, [])
                        if questions:
                            bot_reply = f"根据您的描述，初步推测可能与{result['type_name']}有关。为了更准确的判断，我想再多了解一些情况。{questions[0]}"
                        else:
                            session['chat_state'] = 'report'
                            bot_reply = generate_report(context)
                    else:
                        bot_reply = "抱歉，根据目前的描述，我还无法确定具体的问题方向。能否请您再详细描述一下症状？例如具体的疼痛部位、持续时间、诱发因素等。"
                
                elif state == 'asking_details':
                    # Store answer text
                    context['symptoms_text'] += f"；{user_input}"
                    
                    if context.get('use_dynamic_questions'):
                        # Generate next dynamic question
                        rule_question, _ = get_next_question(context.get('collected_keywords', []))
                        if rule_question:
                            bot_reply = rule_question
                        else:
                            # No more questions or good match found
                            session['chat_state'] = 'report'
                            bot_reply = generate_report(context)
                    else:
                        # Legacy flow
                        type_id = context.get('type_id')
                        q_idx = context.get('question_idx', 0)
                        
                        # Move to next question
                        q_idx += 1
                        context['question_idx'] = q_idx
                        session['chat_context'] = context
                        
                        questions = FOLLOW_UP_QUESTIONS.get(type_id, [])
                        if q_idx < len(questions):
                            bot_reply = questions[q_idx]
                        else:
                            # All questions asked (or round limit reached)
                            session['chat_state'] = 'report'
                            bot_reply = generate_report(context)

                elif state == 'report':
                     bot_reply = "我已经为您生成了报告。如果您有新的问题，请点击页面上方的“重新咨询”按钮。"

            session['chat_history'].append({'role': 'bot', 'content': bot_reply})
            session.modified = True
            
    return render_template('chat.html', history=session['chat_history'])

@app.route('/reset_chat')
def reset_chat():
    session.pop('chat_history', None)
    session.pop('chat_state', None)
    session.pop('chat_context', None)
    session.pop('chat_round', None)
    return redirect(url_for('chat'))

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
