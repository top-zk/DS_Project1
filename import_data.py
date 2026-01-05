import os
import json
from flask import Flask
from models import db, MedicalEncyclopedia
from pypinyin import pinyin, Style

def import_data():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    json_path = os.path.join(os.path.dirname(__file__), 'huank/DB_project/output/diseases.json')
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with app.app_context():
        # Ensure tables exist
        db.create_all()
        
        # Optional: Clear existing data
        # db.session.query(MedicalEncyclopedia).delete()
        # db.session.commit()
        
        print("Starting data import...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            count = 0
            for item in data:
                name = item.get('疾病名称', '').strip()
                if not name:
                    continue
                    
                # Generate Pinyin Index
                py = pinyin(name, style=Style.FIRST_LETTER)
                first_letter = py[0][0][0].upper() if py else '#'
                if not first_letter.isalpha():
                    first_letter = '#'

                # Check if exists
                exists = MedicalEncyclopedia.query.filter_by(disease_name=name).first()
                if exists:
                    # Update existing record
                    exists.symptoms = ", ".join(item.get('主要症状', []))
                    exists.content = item.get('症状描述', '')
                    exists.treatment = item.get('治疗方法', '')
                    exists.causes = item.get('病因', '')
                    exists.prevention = item.get('预防措施', '')
                    exists.department = item.get('症状类型', '')
                    exists.pinyin_index = first_letter
                else:
                    # Create new record
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
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} records...")
                    db.session.commit()
            
            db.session.commit()
            print(f"Import completed successfully. Total records processed: {count}")
            
        except Exception as e:
            print(f"Error during import: {e}")
            db.session.rollback()

if __name__ == "__main__":
    import_data()
