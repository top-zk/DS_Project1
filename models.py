from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
from pypinyin import pinyin, Style

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MedicalEncyclopedia(db.Model):
    __tablename__ = 'encyclopedia'
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(200), nullable=False, index=True) # Mapped from title/疾病名称
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
    if MedicalEncyclopedia.query.first() is None:
        json_path = os.path.join(os.path.dirname(__file__), 'huank/DB_project/output/diseases.json')
        if os.path.exists(json_path):
            print("Loading encyclopedia data from JSON...")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        name = item.get('疾病名称', 'Unknown')
                        # Generate Pinyin Index
                        py = pinyin(name, style=Style.FIRST_LETTER)
                        first_letter = py[0][0][0].upper() if py else '#'
                        if not first_letter.isalpha():
                            first_letter = '#'
                        
                        entry = MedicalEncyclopedia(
                            disease_name=name,
                            symptoms=", ".join(item.get('主要症状', [])),
                            content=item.get('症状描述', ''),
                            treatment=item.get('治疗方法', ''),
                            causes=item.get('病因', ''),
                            prevention=item.get('预防措施', ''),
                            department=item.get('症状类型', ''),
                            pinyin_index=first_letter
                        )
                        db.session.add(entry)
                db.session.commit()
                print("Encyclopedia data loaded.")
            except Exception as e:
                print(f"Error loading encyclopedia data: {e}")
                db.session.rollback()

# Wrapper functions for UI compatibility
def get_encyclopedia_letters():
    # Return list of distinct pinyin_index
    results = db.session.query(MedicalEncyclopedia.pinyin_index).distinct().order_by(MedicalEncyclopedia.pinyin_index).all()
    return [r[0] for r in results]

def get_articles_by_letter(letter):
    return MedicalEncyclopedia.query.filter_by(pinyin_index=letter).order_by(MedicalEncyclopedia.disease_name).all()

def search_articles(query):
    search_term = f'%{query}%'
    return MedicalEncyclopedia.query.filter(
        (MedicalEncyclopedia.disease_name.like(search_term)) | 
        (MedicalEncyclopedia.symptoms.like(search_term))
    ).order_by(MedicalEncyclopedia.disease_name).all()

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
