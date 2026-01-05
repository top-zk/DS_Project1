import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_NAME = "medical_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Encyclopedia table
    c.execute('''
        CREATE TABLE IF NOT EXISTS encyclopedia (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            first_letter TEXT NOT NULL,
            summary TEXT,
            content TEXT,
            symptoms TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Initialize encyclopedia data if empty
    init_encyclopedia_data()

def init_encyclopedia_data():
    conn = get_db_connection()
    c = conn.cursor()
    count = c.execute('SELECT COUNT(*) FROM encyclopedia').fetchone()[0]
    
    if count == 0:
        import json
        json_path = os.path.join(os.path.dirname(__file__), 'huank/DB_project/output/diseases.json')
        if os.path.exists(json_path):
            print("Loading encyclopedia data from JSON...")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        title = item.get('疾病名称', 'Unknown')
                        content = item.get('症状描述', '')
                        symptoms = ", ".join(item.get('主要症状', []))
                        first_letter = title[0].upper() if title else '#'
                        if not first_letter.isalpha():
                            first_letter = '#'
                        
                        c.execute('INSERT INTO encyclopedia (title, first_letter, content, symptoms) VALUES (?, ?, ?, ?)',
                                  (title, first_letter, content, symptoms))
                conn.commit()
                print("Encyclopedia data loaded.")
            except Exception as e:
                print(f"Error loading encyclopedia data: {e}")
        else:
            print(f"Encyclopedia data file not found at {json_path}")
            
    conn.close()

# Encyclopedia Functions
def get_encyclopedia_letters():
    conn = get_db_connection()
    c = conn.cursor()
    rows = c.execute('SELECT DISTINCT first_letter FROM encyclopedia ORDER BY first_letter').fetchall()
    conn.close()
    return [row['first_letter'] for row in rows]

def get_articles_by_letter(letter):
    conn = get_db_connection()
    c = conn.cursor()
    rows = c.execute('SELECT * FROM encyclopedia WHERE first_letter = ? ORDER BY title', (letter,)).fetchall()
    conn.close()
    return rows

def search_articles(query):
    conn = get_db_connection()
    c = conn.cursor()
    search_term = f'%{query}%'
    rows = c.execute('SELECT * FROM encyclopedia WHERE title LIKE ? OR symptoms LIKE ? ORDER BY title', 
                     (search_term, search_term)).fetchall()
    conn.close()
    return rows

def get_article_by_id(article_id):
    conn = get_db_connection()
    c = conn.cursor()
    row = c.execute('SELECT * FROM encyclopedia WHERE id = ?', (article_id,)).fetchone()
    conn.close()
    return row

def add_user(email, password, first_name, last_name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        password_hash = generate_password_hash(password)
        print(f"DEBUG: Attempting to insert user {email}...")
        c.execute('INSERT INTO users (email, password_hash, first_name, last_name) VALUES (?, ?, ?, ?)',
                  (email, password_hash, first_name, last_name))
        conn.commit()
        print(f"DEBUG: User {email} inserted successfully.")
        return True
    except sqlite3.IntegrityError as e:
        print(f"DEBUG: IntegrityError for {email}: {e}")
        return False
    except Exception as e:
        print(f"DEBUG: Unexpected error adding user {email}: {e}")
        return False
    finally:
        try:
            conn.close()
        except:
            pass

def get_user_by_email(email):
    conn = get_db_connection()
    c = conn.cursor()
    user = c.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    user = c.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def verify_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user['password_hash'], password):
        return user
    return None
