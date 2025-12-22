import sqlite3
import json

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('medical_data.db')
cursor = conn.cursor()

# Create a table to store the disease data
cursor.execute('''
CREATE TABLE IF NOT EXISTS diseases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_name TEXT,
    source_url TEXT,
    summary TEXT,
    primary_symptoms TEXT,
    secondary_symptoms TEXT,
    examination_indicators TEXT,
    diagnostic_criteria TEXT,
    differential_diagnoses TEXT
)
''')

# Load the data from the JSON file
with open('pneumonia.json', 'r', encoding='utf-8') as f:
    disease_data = json.load(f)

# Insert the data into the table
cursor.execute("""
INSERT INTO diseases (disease_name, source_url, summary, primary_symptoms, secondary_symptoms, examination_indicators, diagnostic_criteria, differential_diagnoses)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    disease_data['疾病名称'],
    disease_data['来源'],
    disease_data['摘要'],
    json.dumps(disease_data['主要症状']),
    json.dumps(disease_data['次要症状']),
    json.dumps(disease_data['检查指标']),
    disease_data['诊断标准'],
    json.dumps(disease_data['鉴别诊断'])
))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data inserted into the database successfully.")