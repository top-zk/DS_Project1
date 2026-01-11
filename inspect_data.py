import json
import os
from collections import Counter

file_path = r"huank/DB_project/output/diseases.json"

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    categories = []
    for item in data:
        cat = item.get('症状类型')
        categories.append(str(cat))
    
    counts = Counter(categories)
    print("\nCategory Counts:")
    for cat, count in counts.most_common(20):
        print(f"{cat}: {count}")
else:
    print("File not found")
