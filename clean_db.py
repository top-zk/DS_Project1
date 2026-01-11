from medical_ui import app
from models import db, MedicalEncyclopedia
import re

def clean_content():
    with app.app_context():
        # Find entries with script tags
        # Regex for <script>...</script> (case insensitive, multiline)
        script_content_pattern = re.compile(r'<script\b[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        
        # Regex for single <script ...> tag
        script_tag_pattern = re.compile(r'<script\b[^>]*>', re.IGNORECASE)
        
        # Regex for </script> tag
        script_close_pattern = re.compile(r'</script>', re.IGNORECASE)

        # Get all entries that might contain "script"
        entries = MedicalEncyclopedia.query.filter(MedicalEncyclopedia.content.like('%script%')).all()
        
        count = 0
        for entry in entries:
            original = entry.content
            cleaned = script_content_pattern.sub('', original)
            cleaned = script_tag_pattern.sub('', cleaned)
            cleaned = script_close_pattern.sub('', cleaned)
            
            if original != cleaned:
                entry.content = cleaned
                count += 1
                print(f"Cleaned entry: {entry.title}")
        
        if count > 0:
            db.session.commit()
            print(f"Successfully cleaned {count} entries.")
        else:
            print("No entries needed cleaning (found entries only contained 'script' as part of other words like 'prescription').")

if __name__ == '__main__':
    clean_content()
