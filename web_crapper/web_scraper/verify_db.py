import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('medical_data.db')
cursor = conn.cursor()

# Select all data from the diseases table
cursor.execute("SELECT * FROM diseases")

# Fetch all the rows
rows = cursor.fetchall()

print("--- 内容来自 medical_data.db 数据库 ---")
# Print the data
for row in rows:
    print("ID:", row[0])
    print("疾病名称:", row[1])
    print("来源:", row[2])
    print("摘要:", row[3])
    # The following fields are stored as JSON strings, so we load them back
    print("主要症状:", json.loads(row[4]))
    print("次要症状:", json.loads(row[5]))
    print("检查指标:", json.loads(row[6]))
    print("诊断标准:", row[7])
    print("鉴别诊断:", json.loads(row[8]))
    print("-" * 20)

# Close the connection
conn.close()