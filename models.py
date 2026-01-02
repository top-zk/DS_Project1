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
    conn.commit()
    conn.close()

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
