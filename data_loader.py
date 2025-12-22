# data_loader.py
import os
import re
import json
import pandas as pd
import numpy as np
from config import DATA_PATH, SAMPLE_DATA, TRAIN_DATA_PATH, VAL_DATA_PATH, ENCYCLOPEDIA_JSON_DIR
from sklearn.model_selection import train_test_split
import subprocess
import importlib


def install_required_packages():
    """安装必要的Python包"""
    required = ['xlrd', 'openpyxl']
    for package in required:
        try:
            importlib.import_module(package)
            print(f"{package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call(['pip', 'install', package])
            print(f"{package} 安装完成")


def load_data_from_excel(file_path):
    """从Excel文件加载数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []

    try:
        # 根据文件扩展名选择引擎
        ext = os.path.splitext(file_path)[1].lower()
        df = None

        # 尝试不同引擎
        if ext == '.xlsx':
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                df = pd.read_excel(file_path)  # 尝试默认引擎
        elif ext == '.xls':
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except:
                df = pd.read_excel(file_path)  # 尝试默认引擎
        else:
            df = pd.read_excel(file_path)  # 其他格式使用默认引擎

        print(f"成功读取Excel文件: {file_path}")

        # 提取业务问题
        op_questions = []
        ci_questions = []
        tir_questions = []
        cr_questions = []
        cs_questions = []
        rq_questions = []
        sales_questions = []
        pay_questions = []
        sv_questions = []
        store_questions = []

        # 检查列名
        if '油价问题' in df.columns and '商品销量问题' in df.columns and '拐入率问题' in df.columns and '转化率问题' in df.columns and '卡销问题' in df.columns and '油枪号码问题' in df.columns and '油品单量问题' in df.columns and '支付问题' in df.columns and '销售额问题' in df.columns and '库存问题' in df.columns:
            for idx, row in df.iterrows():
                # 使用列名访问
                if pd.notna(row['油价问题']) and re.search(r'[^\s]', str(row['油价问题'])):
                    op_questions.append((str(row['油价问题']).strip(), 0))
                if pd.notna(row['商品销量问题']) and re.search(r'[^\s]', str(row['商品销量问题'])):
                    ci_questions.append((str(row['商品销量问题']).strip(), 1))
                if pd.notna(row['拐入率问题']) and re.search(r'[^\s]', str(row['拐入率问题'])):
                    tir_questions.append((str(row['拐入率问题']).strip(), 2))
                if pd.notna(row['转化率问题']) and re.search(r'[^\s]', str(row['转化率问题'])):
                    cr_questions.append((str(row['转化率问题']).strip(), 3))
                if pd.notna(row['卡销问题']) and re.search(r'[^\s]', str(row['卡销问题'])):
                    cs_questions.append((str(row['卡销问题']).strip(), 4))
                if pd.notna(row['油枪号码问题']) and re.search(r'[^\s]', str(row['油枪号码问题'])):
                    rq_questions.append((str(row['油枪号码问题']).strip(), 5))
                if pd.notna(row['油品单量问题']) and re.search(r'[^\s]', str(row['油品单量问题'])):
                    sales_questions.append((str(row['油品单量问题']).strip(), 6))
                if pd.notna(row['支付问题']) and re.search(r'[^\s]', str(row['支付问题'])):
                    pay_questions.append((str(row['支付问题']).strip(), 7))
                if pd.notna(row['销售额问题']) and re.search(r'[^\s]', str(row['销售额问题'])):
                    sv_questions.append((str(row['销售额问题']).strip(), 8))
                if pd.notna(row['库存问题']) and re.search(r'[^\s]', str(row['库存问题'])):
                    store_questions.append((str(row['库存问题']).strip(), 9))
        else:
            # 使用位置索引作为备选方案
            if len(df.columns) < 3:
                print("Excel文件列数不足，需要至少3列")
                return []

            for idx, row in df.iterrows():
                if len(row) > 1 and pd.notna(row.iloc[1]) and re.search(r'[^\s]', str(row.iloc[1])):
                    op_questions.append((str(row.iloc[1]).strip(), 0))
                if len(row) > 2 and pd.notna(row.iloc[2]) and re.search(r'[^\s]', str(row.iloc[2])):
                    ci_questions.append((str(row.iloc[2]).strip(), 1))
                if len(row) > 3 and pd.notna(row.iloc[3]) and re.search(r'[^\s]', str(row.iloc[3])):
                    tir_questions.append((str(row.iloc[3]).strip(), 2))
                if len(row) > 4 and pd.notna(row.iloc[4]) and re.search(r'[^\s]', str(row.iloc[4])):
                    cr_questions.append((str(row.iloc[4]).strip(), 3))
                if len(row) > 5 and pd.notna(row.iloc[5]) and re.search(r'[^\s]', str(row.iloc[5])):
                    cs_questions.append((str(row.iloc[5]).strip(), 4))
                if len(row) > 6 and pd.notna(row.iloc[6]) and re.search(r'[^\s]', str(row.iloc[6])):
                    rq_questions.append((str(row.iloc[6]).strip(), 5))
                if len(row) > 7 and pd.notna(row.iloc[7]) and re.search(r'[^\s]', str(row.iloc[7])):
                    sales_questions.append((str(row.iloc[7]).strip(), 6))
                if len(row) > 8 and pd.notna(row.iloc[8]) and re.search(r'[^\s]', str(row.iloc[8])):
                    pay_questions.append((str(row.iloc[8]).strip(), 7))
                if len(row) > 9 and pd.notna(row.iloc[9]) and re.search(r'[^\s]', str(row.iloc[9])):
                    sv_questions.append((str(row.iloc[9]).strip(), 8))
                if len(row) > 10 and pd.notna(row.iloc[10]) and re.search(r'[^\s]', str(row.iloc[10])):
                    store_questions.append((str(row.iloc[10]).strip(), 9))

        return (op_questions + ci_questions + tir_questions + cr_questions +
                cs_questions + rq_questions + sales_questions +
                pay_questions + sv_questions + store_questions)

    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return []


def save_data_to_txt(data, file_path):
    """将数据保存到文本文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for text, label in data:
            f.write(f"{text}\t{label}\n")
    print(f"已保存 {len(data)} 条数据到 {file_path}")


def load_data_from_txt(file_path):
    """从文本文件加载数据"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                text = parts[0]
                try:
                    label = int(parts[1])
                    data.append((text, label))
                except ValueError:
                    continue
    print(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data

def load_data_from_json_file(file_path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        items = obj if isinstance(obj, list) else obj.get('data', [])
        if not items and isinstance(obj, dict):
            text = obj.get('text') or obj.get('内容') or obj.get('question') or obj.get('症状描述') or obj.get('content')
            if not text:
                t1 = obj.get('summary') or ''
                t2 = obj.get('title') or ''
                text = (str(t1) + ' ' + str(t2)).strip()
            label = obj.get('label') or obj.get('类别') or obj.get('症状类型') or obj.get('category')
            if label is None:
                label = 0
            try:
                label = int(label)
            except Exception:
                label = 0
            t = str(text).strip()
            if t:
                result.append((t, label))
        else:
            for it in items:
                text = it.get('text') or it.get('内容') or it.get('question') or it.get('症状描述') or it.get('content')
                if not text:
                    t1 = it.get('summary') or ''
                    t2 = it.get('title') or ''
                    text = (str(t1) + ' ' + str(t2)).strip()
                label = it.get('label') or it.get('类别') or it.get('症状类型') or it.get('category')
                if label is None:
                    label = 0
                try:
                    label = int(label)
                except Exception:
                    label = 0
                t = str(text).strip()
                if t:
                    result.append((t, label))
    except Exception:
        pass
    return result

def load_data_from_json_directory(dir_path):
    data = []
    if not os.path.isdir(dir_path):
        return data
    for root, _, files in os.walk(dir_path):
        for name in files:
            if name.lower().endswith('.json'):
                data.extend(load_data_from_json_file(os.path.join(root, name)))
    return data


def prepare_data():
    """准备数据并保存为文本文件"""
    install_required_packages()
    # 确保data文件夹存在（已在config中创建，这里可省略）
    # 但为了更健壮，可以在这里再确认一次
    from config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    data = []
    json_data = load_data_from_json_directory(ENCYCLOPEDIA_JSON_DIR)
    if json_data:
        data = json_data
    elif os.path.exists(DATA_PATH):
        data = load_data_from_excel(DATA_PATH)
    else:
        data = SAMPLE_DATA

    # 确保有足够的数据
    if not data:
        print("数据加载失败，使用样本数据")
        data = SAMPLE_DATA

    # 划分数据集
    if len(data) < 5:
        print("数据量较少，使用全部数据作为训练集")
        train_data = data
        val_data = []
    else:
        texts = [item[0] for item in data]
        labels = [item[1] for item in data]
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        can_strat = all(c >= 2 for c in counts.values())
        if can_strat:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            train_data = list(zip(train_texts, train_labels))
            val_data = list(zip(val_texts, val_labels))
        else:
            train_data = data
            val_data = []

    # 保存数据
    save_data_to_txt(train_data, TRAIN_DATA_PATH)
    if val_data:
        save_data_to_txt(val_data, VAL_DATA_PATH)

    print(f"训练集: {len(train_data)}条")
    print(f"验证集: {len(val_data)}条" if val_data else "无验证集")


if __name__ == "__main__":
    prepare_data()

