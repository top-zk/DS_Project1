import os
import json
import glob
import logging
import time
from datetime import datetime
from pypinyin import pinyin, Style
from flask import Flask
from sqlalchemy import text
from models import db, MedicalEncyclopedia
from medical_ui import app

# Setup logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"import_data_{timestamp}.log")
report_file = os.path.join(log_dir, f"import_report_{timestamp}.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def verify_data_integrity(json_files):
    logging.info("Step 1: Verifying data integrity...")
    valid_items = 0
    invalid_items = 0
    files_status = {}

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    items = data['data']
                elif 'title' in data or '疾病名称' in data:
                    items = [data]
            
            file_valid = 0
            file_invalid = 0
            for item in items:
                name = item.get('疾病名称') or item.get('title')
                if name and str(name).strip():
                    file_valid += 1
                else:
                    file_invalid += 1
            
            valid_items += file_valid
            invalid_items += file_invalid
            files_status[json_path] = {"valid": file_valid, "invalid": file_invalid, "status": "OK"}
            logging.info(f"Verified {json_path}: {file_valid} valid, {file_invalid} invalid.")

        except json.JSONDecodeError:
            logging.error(f"Invalid JSON format in {json_path}")
            files_status[json_path] = {"status": "Corrupt JSON", "valid": 0, "invalid": 0}
        except Exception as e:
            logging.error(f"Error reading {json_path}: {e}")
            files_status[json_path] = {"status": f"Error: {str(e)}", "valid": 0, "invalid": 0}

    return valid_items, invalid_items, files_status

def check_system_readiness():
    logging.info("Step 2: Checking system readiness...")
    try:
        # Check DB connection
        db.session.execute(text("SELECT 1"))
        logging.info("Database connection established.")
        return True
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return False

def import_data():
    start_time = time.time()
    logging.info("Starting Data Import Process")

    with app.app_context():
        # 1. Discovery
        search_dirs = [
            'output',
            os.path.join('huank', 'DB_project', 'output'),
            os.path.join('huank', 'DB_project')
        ]
        
        json_files = []
        for d in search_dirs:
            if os.path.exists(d):
                files = glob.glob(os.path.join(d, '*.json'))
                json_files.extend(files)
        
        # Filter relevant files
        json_files = list(set([f for f in json_files if not any(x in f for x in ['package', 'lock', 'schema', 'config'])]))
        
        if not json_files:
            logging.warning("No data files found.")
            return

        # 2. Verification
        total_valid_source, total_invalid_source, files_verification = verify_data_integrity(json_files)
        
        # 3. Readiness
        if not check_system_readiness():
            logging.error("System not ready. Aborting.")
            return

        # Pre-import count
        pre_count = MedicalEncyclopedia.query.count()
        logging.info(f"Database records before import: {pre_count}")

        # 4. Execution
        logging.info("Step 3: Executing import...")
        total_inserted = 0
        total_updated = 0
        errors = []

        for json_path, status in files_verification.items():
            if status['status'] != "OK" or status['valid'] == 0:
                continue

            logging.info(f"Importing from {json_path}...")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Normalize (logic repeated but safe)
                items = data if isinstance(data, list) else (data.get('data', []) if isinstance(data, dict) else [data])
                
                for item in items:
                    try:
                        raw_name = item.get('疾病名称') or item.get('title')
                        if not raw_name:
                            continue
                            
                        name = raw_name.strip()
                        
                        # Generate Pinyin
                        py = pinyin(name, style=Style.FIRST_LETTER)
                        first_letter = py[0][0][0].upper() if py else '#'
                        if not first_letter.isalpha():
                            first_letter = '#'
                        
                        # Map Fields
                        symptoms_val = item.get('主要症状') or item.get('symptoms') or []
                        symptoms_str = ", ".join([str(s) for s in symptoms_val]) if isinstance(symptoms_val, list) else str(symptoms_val)
                        
                        content = item.get('症状描述') or item.get('content') or item.get('summary') or ''
                        treatment = str(item.get('治疗方法') or item.get('treatment') or '')
                        causes = str(item.get('病因') or item.get('causes') or '')
                        prevention = str(item.get('预防措施') or item.get('prevention') or '')
                        department = item.get('症状类型') or item.get('department') or ''
                        
                        # Upsert
                        existing = MedicalEncyclopedia.query.filter_by(disease_name=name).first()
                        if existing:
                            existing.symptoms = symptoms_str
                            existing.content = content
                            existing.treatment = treatment
                            existing.causes = causes
                            existing.prevention = prevention
                            existing.department = department
                            existing.pinyin_index = first_letter
                            total_updated += 1
                        else:
                            new_entry = MedicalEncyclopedia(
                                disease_name=name,
                                symptoms=symptoms_str,
                                content=content,
                                treatment=treatment,
                                causes=causes,
                                prevention=prevention,
                                department=department,
                                pinyin_index=first_letter
                            )
                            db.session.add(new_entry)
                            total_inserted += 1
                            
                    except Exception as e:
                        errors.append({"file": json_path, "item": name, "error": str(e)})
                
                db.session.commit()
                
            except Exception as e:
                logging.error(f"Failed to process file {json_path}: {e}")
                errors.append({"file": json_path, "error": str(e)})
                db.session.rollback()

        # 5. Consistency Check
        post_count = MedicalEncyclopedia.query.count()
        logging.info(f"Step 4: Consistency Check - Database records after import: {post_count}")
        logging.info(f"Net change: {post_count - pre_count} (Expected new: {total_inserted})")
        
        end_time = time.time()
        duration = end_time - start_time

        # 6. Report Generation
        report = {
            "timestamp": timestamp,
            "duration_seconds": round(duration, 2),
            "files_processed": len(files_verification),
            "source_valid_items": total_valid_source,
            "db_pre_count": pre_count,
            "db_post_count": post_count,
            "inserted": total_inserted,
            "updated": total_updated,
            "errors_count": len(errors),
            "errors_details": errors[:10]  # Top 10 errors
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Step 5: Import Complete. Report saved to {report_file}")
        print(f"\nImport Summary:")
        print(f"  Total Processed: {total_inserted + total_updated}")
        print(f"  Inserted: {total_inserted}")
        print(f"  Updated: {total_updated}")
        print(f"  Errors: {len(errors)}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Full report: {report_file}")

if __name__ == '__main__':
    import_data()
