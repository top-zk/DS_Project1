from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
from pypinyin import pinyin, Style
from sqlalchemy import or_, case, func

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    query_text = db.Column(db.Text, nullable=False)
    prediction_result = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.String(50))  # 'correct', 'incorrect'

class MedicalEncyclopedia(db.Model):
    __tablename__ = 'encyclopedia'
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(200), nullable=False, index=True, unique=True) # Mapped from title/疾病名称
    symptoms = db.Column(db.Text) # Mapped from symptoms/主要症状
    content = db.Column(db.Text) # Mapped from content/症状描述
    treatment = db.Column(db.Text) # Mapped from treatment/治疗方法
    causes = db.Column(db.Text) # Mapped from causes/病因
    prevention = db.Column(db.Text) # Mapped from prevention/预防措施
    department = db.Column(db.String(100)) # Mapped from 症状类型
    pinyin_index = db.Column(db.String(10), index=True) # A-Z index

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.disease_name, # Maintain compatibility with templates using 'title'
            'symptoms': self.symptoms,
            'content': self.content,
            'treatment': self.treatment,
            'causes': self.causes,
            'prevention': self.prevention,
            'department': self.department,
            'first_letter': self.pinyin_index # Maintain compatibility with templates using 'first_letter'
        }

    # Compatibility properties for templates
    @property
    def title(self):
        return self.disease_name
    
    @property
    def first_letter(self):
        return self.pinyin_index

# Legacy support functions (to be refactored or kept as wrappers)
# We will use SQLAlchemy for new operations but keep these if needed or rewrite them to use SQLAlchemy

def init_db(app):
    # db.init_app(app) # Should be called in main app
    with app.app_context():
        db.create_all()
        init_encyclopedia_data()

def init_encyclopedia_data():
    # Check for duplicates to fix existing issues
    try:
        dupes = db.session.query(MedicalEncyclopedia.disease_name, func.count(MedicalEncyclopedia.disease_name))\
            .group_by(MedicalEncyclopedia.disease_name)\
            .having(func.count(MedicalEncyclopedia.disease_name) > 1).all()
        
        if dupes:
            print(f"Detected {len(dupes)} duplicate entries. Clearing encyclopedia table to reload...")
            MedicalEncyclopedia.query.delete()
            db.session.commit()
    except Exception as e:
        print(f"Error checking duplicates: {e}")
        # If column doesn't exist or other error, might need to drop table
        # We'll continue and see if we can load.

    if MedicalEncyclopedia.query.first() is None:
        # Load from multiple sources
        sources = [
            os.path.join(os.path.dirname(__file__), 'huank/DB_project/demo_output.json'), # High quality Chinese data
            os.path.join(os.path.dirname(__file__), 'huank/DB_project/output/diseases.json') # Original data (mixed)
        ]
        
        # Load existing names to avoid duplicates (in case of partial load)
        existing_names = set(n.lower().strip() for n in db.session.query(MedicalEncyclopedia.disease_name).all())
        loaded_count = 0
        
        for json_path in sources:
            if os.path.exists(json_path):
                print(f"Loading encyclopedia data from {json_path}...")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle different formats (list or dict)
                        items = data if isinstance(data, list) else data.get('data', [])
                        
                        for item in items:
                            raw_name = item.get('疾病名称') or item.get('title')
                            if not raw_name:
                                continue
                            
                            # Normalize name
                            name = raw_name.strip()
                            norm_name = name.lower()
                            
                            if norm_name in existing_names:
                                continue
                                
                            existing_names.add(norm_name)
                            
                            # Generate Pinyin Index
                            py = pinyin(name, style=Style.FIRST_LETTER)
                            first_letter = py[0][0][0].upper() if py else '#'
                            if not first_letter.isalpha():
                                first_letter = '#'
                            
                            # Get symptoms safely
                            symptoms_val = item.get('主要症状') or item.get('symptoms') or []
                            if isinstance(symptoms_val, list):
                                symptoms_str = ", ".join([str(s) for s in symptoms_val])
                            else:
                                symptoms_str = str(symptoms_val)
                                
                            entry = MedicalEncyclopedia(
                                disease_name=name,
                                symptoms=symptoms_str,
                                content=item.get('症状描述') or item.get('content') or item.get('summary') or '',
                                treatment=item.get('治疗方法') or item.get('treatment') or '',
                                causes=item.get('病因') or item.get('causes') or '',
                                prevention=item.get('预防措施') or item.get('prevention') or '',
                                department=item.get('症状类型') or item.get('department') or '',
                                pinyin_index=first_letter
                            )
                            db.session.add(entry)
                            loaded_count += 1
                            
                    db.session.commit()
                    print(f"Loaded {loaded_count} items from {json_path}.")
                    loaded_count = 0 # Reset for next source report
                except Exception as e:
                    print(f"Error loading encyclopedia data from {json_path}: {e}")
                    db.session.rollback()

# Wrapper functions for UI compatibility
def get_encyclopedia_letters():
    # Return list of distinct pinyin_index
    results = db.session.query(MedicalEncyclopedia.pinyin_index).distinct().order_by(MedicalEncyclopedia.pinyin_index).all()
    return [r[0] for r in results]

def get_articles_by_letter(letter, page=1, per_page=10):
    return MedicalEncyclopedia.query.filter_by(pinyin_index=letter)\
        .order_by(MedicalEncyclopedia.disease_name)\
        .paginate(page=page, per_page=per_page, error_out=False)

def search_articles(query, page=1, per_page=10):
    if not query:
        return None
    
    # Split query into keywords
    keywords = query.strip().split()
    if not keywords:
        return None
        
    # Build OR conditions for name and symptoms
    conditions = []
    for kw in keywords:
        term = f'%{kw}%'
        conditions.append(MedicalEncyclopedia.disease_name.like(term))
        conditions.append(MedicalEncyclopedia.symptoms.like(term))
    
    # Custom sorting: Prioritize matches in disease_name
    # We construct a case statement: 1 if name matches first keyword, else 2
    # Simplified: Name matches any keyword -> higher priority
    name_match_conditions = [MedicalEncyclopedia.disease_name.like(f'%{kw}%') for kw in keywords]
    
    return MedicalEncyclopedia.query.filter(or_(*conditions))\
        .order_by(
            case((or_(*name_match_conditions), 0), else_=1), # 0 comes before 1
            MedicalEncyclopedia.disease_name
        )\
        .paginate(page=page, per_page=per_page, error_out=False)

def get_related_diseases(article_id):
    article = MedicalEncyclopedia.query.get(article_id)
    if not article or not article.department:
        return []
    
    # Get 5 random diseases from same department, excluding current
    return MedicalEncyclopedia.query.filter(
        MedicalEncyclopedia.department == article.department,
        MedicalEncyclopedia.id != article_id
    ).order_by(func.random()).limit(5).all()

def get_article_by_id(article_id):
    return MedicalEncyclopedia.query.get(article_id)

def add_user(email, password, first_name, last_name):
    try:
        password_hash = generate_password_hash(password)
        new_user = User(email=email, password_hash=password_hash, first_name=first_name, last_name=last_name)
        db.session.add(new_user)
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        db.session.rollback()
        return False

def verify_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password_hash, password):
        return {
            'id': user.id,
            'email': user.email,
            'first_name': user.first_name,
            'password_hash': user.password_hash
        }
    return None

def log_interaction(user_id, query, result):
    try:
        interaction = UserInteraction(
            user_id=user_id,
            query_text=query,
            prediction_result=json.dumps(result, ensure_ascii=False)
        )
        db.session.add(interaction)
        db.session.commit()
        return True
    except Exception as e:
        print(f"Error logging interaction: {e}")
        db.session.rollback()
        return False
