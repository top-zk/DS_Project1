import os
import re
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from medical_config import DATA_PATH, SAMPLE_DATA, TRAIN_DATA_PATH, VAL_DATA_PATH, DISEASE_SYMPTOM_TYPES, DATA_DIR, ENCYCLOPEDIA_DIR, ENCYCLOPEDIA_JSON_DIR, DISEASE_JSON_PATH, DISEASE_XML_PATH
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


def load_disease_data_from_excel(file_path):
    """从Excel文件加载疾病症状数据"""
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

        print(f"成功读取疾病数据文件: {file_path}")

        disease_data = []

        # 检查列名 - 更新为疾病数据列名
        if ('疾病名称' in df.columns and '主要症状' in df.columns and
                '症状描述' in df.columns and '症状类型' in df.columns):

            for idx, row in df.iterrows():
                if (pd.notna(row['疾病名称']) and pd.notna(row['症状描述']) and
                        pd.notna(row['症状类型']) and re.search(r'[^\s]', str(row['症状描述']))):

                    try:
                        disease_name = str(row['疾病名称']).strip()
                        symptom_description = str(row['症状描述']).strip()
                        symptom_type = int(row['症状类型'])

                        # 处理主要症状
                        main_symptoms = []
                        if pd.notna(row['主要症状']):
                            if isinstance(row['主要症状'], str):
                                main_symptoms = [s.strip() for s in row['主要症状'].split('，') if s.strip()]

                        disease_data.append({
                            "疾病名称": disease_name,
                            "主要症状": main_symptoms,
                            "症状描述": symptom_description,
                            "症状类型": symptom_type
                        })
                    except (ValueError, KeyError) as e:
                        print(f"处理行 {idx} 时出错: {e}")
                        continue

        else:
            # 使用位置索引作为备选方案
            if len(df.columns) < 4:
                print("疾病数据文件列数不足，需要至少4列")
                return []

            for idx, row in df.iterrows():
                if (len(row) > 1 and pd.notna(row.iloc[1]) and
                        len(row) > 2 and pd.notna(row.iloc[2]) and
                        len(row) > 3 and pd.notna(row.iloc[3]) and
                        re.search(r'[^\s]', str(row.iloc[2]))):

                    try:
                        disease_name = str(row.iloc[1]).strip()
                        symptom_description = str(row.iloc[2]).strip()
                        symptom_type = int(row.iloc[3])

                        disease_data.append({
                            "疾病名称": disease_name,
                            "主要症状": [],
                            "症状描述": symptom_description,
                            "症状类型": symptom_type
                        })
                    except (ValueError, IndexError) as e:
                        print(f"处理行 {idx} 时出错: {e}")
                        continue

        print(f"成功加载 {len(disease_data)} 条疾病数据")
        return disease_data

    except Exception as e:
        print(f"读取疾病数据文件失败: {e}")
        return []


def save_disease_data_to_txt(data, file_path):
    """将疾病数据保存到文本文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if isinstance(item, dict) and '症状描述' in item and '症状类型' in item:
                f.write(f"{item['症状描述']}\t{item['症状类型']}\n")
            elif isinstance(item, tuple) and len(item) == 2:
                f.write(f"{item[0]}\t{item[1]}\n")
    print(f"已保存 {len(data)} 条疾病数据到 {file_path}")


def load_disease_data_from_txt(file_path):
    """从文本文件加载疾病数据"""
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
    print(f"从 {file_path} 加载了 {len(data)} 条疾病数据")
    return data

def load_disease_data_from_directory(dir_path):
    if not os.path.isdir(dir_path):
        return []
    name_to_id = {v: k for k, v in DISEASE_SYMPTOM_TYPES.items()}
    data = []
    for entry in os.scandir(dir_path):
        if entry.is_dir():
            label = None
            name = os.path.basename(entry.path)
            if name.isdigit():
                try:
                    label = int(name)
                except:
                    label = None
            elif name in name_to_id:
                label = name_to_id[name]
            if label is None:
                continue
            for f in os.scandir(entry.path):
                if f.is_file() and f.name.lower().endswith('.txt'):
                    with open(f.path, 'r', encoding='utf-8') as fp:
                        for line in fp:
                            t = line.strip()
                            t = clean_text(t)
                            if t:
                                data.append((t, label))
    return data

def clean_text(t):
    if not t:
        return t
    s = str(t)
    s = re.sub(r"http[s]?://\S+", "", s)
    s = re.sub(r"(来源|网页源|Source|URL|链接)[:：].*", "", s)
    s = re.sub(r"(爬取时间|采集时间|抓取时间|时间|Time)[:：].*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_disease_data_from_json(file_path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        items = obj if isinstance(obj, list) else obj.get('data', [])
        name_to_id = {v: k for k, v in DISEASE_SYMPTOM_TYPES.items()}
        if not items and isinstance(obj, dict):
            text = obj.get('症状描述') or obj.get('text') or obj.get('symptom') or obj.get('content')
            if not text:
                t1 = obj.get('summary') or ''
                t2 = obj.get('title') or ''
                text = (str(t1) + ' ' + str(t2)).strip()
            label_val = obj.get('症状类型') or obj.get('label') or obj.get('category')
            if isinstance(label_val, str):
                label = name_to_id.get(label_val, 0)
            else:
                label = label_val if label_val is not None else 0
            try:
                label = int(label)
            except:
                label = 0
            t = str(text).strip()
            t = clean_text(t)
            if t:
                result.append((t, label))
        else:
            for it in items:
                text = it.get('症状描述') or it.get('text') or it.get('symptom') or it.get('content')
                if not text:
                    t1 = it.get('summary') or ''
                    t2 = it.get('title') or ''
                    text = (str(t1) + ' ' + str(t2)).strip()
                label_val = it.get('症状类型') or it.get('label') or it.get('category')
                if isinstance(label_val, str):
                    label = name_to_id.get(label_val, 0)
                else:
                    label = label_val if label_val is not None else 0
                try:
                    label = int(label)
                except:
                    label = 0
                t = str(text).strip()
                t = clean_text(t)
                if t:
                    result.append((t, label))
    except Exception:
        pass
    return result

def normalize_category(name):
    if not name:
        return None
    s = str(name).strip()
    mapping = {
        '呼吸系统': '呼吸系统疾病',
        '心血管系统': '心血管系统疾病',
        '消化系统': '消化系统疾病',
        '神经系统': '神经系统疾病',
        '肌肉骨骼': '肌肉骨骼疾病',
        '皮肤': '皮肤疾病',
        '泌尿系统': '泌尿系统疾病',
        '全身性': '全身性疾病',
        '五官': '五官疾病',
        '精神心理': '精神心理疾病'
    }
    if s in mapping:
        return mapping[s]
    if s.endswith('疾病'):
        return s
    return None

def load_disease_pairs_from_diseases_json(file_path):
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        name_to_id = {v: k for k, v in DISEASE_SYMPTOM_TYPES.items()}
        for it in arr if isinstance(arr, list) else []:
            text = it.get('症状描述') or it.get('text') or it.get('summary') or ''
            text = clean_text(str(text).strip())
            cat = it.get('症状类型') or it.get('category')
            catn = normalize_category(cat)
            if not catn:
                continue
            label = name_to_id.get(catn)
            if label is None:
                continue
            if text:
                pairs.append((text, int(label)))
    except Exception:
        pass
    return pairs

def save_disease_data_to_xml(data, file_path):
    root = ET.Element('dataset')
    for text, label in data:
        item = ET.SubElement(root, 'item')
        ET.SubElement(item, 'text').text = str(text)
        ET.SubElement(item, 'label').text = str(int(label))
    tree = ET.ElementTree(root)
    dirn = os.path.dirname(file_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    tree.write(file_path, encoding='utf-8', xml_declaration=True)

def load_disease_data_from_xml(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        result = []
        for item in root.findall('item'):
            te = item.find('text')
            le = item.find('label')
            if te is not None and le is not None:
                t = (te.text or '').strip()
                try:
                    l = int(le.text)
                except:
                    continue
                if t:
                    result.append((t, l))
        return result
    except Exception:
        return []


def prepare_disease_data():
    """准备疾病数据并保存为文本文件"""
    install_required_packages()
    os.makedirs(DATA_DIR, exist_ok=True)

    data = []
    # 1) 优先从 disease.xml 读取
    if os.path.exists(DISEASE_XML_PATH):
        data = load_disease_data_from_xml(DISEASE_XML_PATH)
    # 2) 若无 disease.xml，则从 diseases.json 提取并生成 disease.xml 后再读取
    if not data:
        disease_json_pairs = load_disease_pairs_from_diseases_json(DISEASE_JSON_PATH) if os.path.exists(DISEASE_JSON_PATH) else []
        if disease_json_pairs:
            save_disease_data_to_xml(disease_json_pairs, DISEASE_XML_PATH)
            xml_loaded = load_disease_data_from_xml(DISEASE_XML_PATH)
            data = xml_loaded or disease_json_pairs
    # 3) 若仍为空，回退到百科目录与JSON
    if not data:
        txt_data = []
        if os.path.isdir(ENCYCLOPEDIA_DIR):
            txt_data = load_disease_data_from_directory(ENCYCLOPEDIA_DIR)
        json_data = []
        if os.path.isdir(ENCYCLOPEDIA_JSON_DIR):
            for r, _, files in os.walk(ENCYCLOPEDIA_JSON_DIR):
                for n in files:
                    if n.lower().endswith('.json'):
                        json_data.extend(load_disease_data_from_json(os.path.join(r, n)))
        if json_data:
            xml_path = os.path.join(DATA_DIR, 'encyclopedia_data.xml')
            save_disease_data_to_xml(json_data, xml_path)
            xml_loaded = load_disease_data_from_xml(xml_path)
            data = (txt_data or []) + (xml_loaded or json_data)
        else:
            data = txt_data
    if not data and os.path.exists(DATA_PATH):
        disease_data = load_disease_data_from_excel(DATA_PATH)
        data = [(item['症状描述'], item['症状类型']) for item in disease_data]
    if not data:
        data = [(item['症状描述'], item['症状类型']) for item in SAMPLE_DATA]

    if not data:
        data = [(item['症状描述'], item['症状类型']) for item in SAMPLE_DATA]

    # 划分数据集
    if len(data) < 5:
        print("疾病数据量较少，使用全部数据作为训练集")
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
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            train_data = list(zip(train_texts, train_labels))
            val_data = list(zip(val_texts, val_labels))

    # 保存数据
    save_disease_data_to_txt(train_data, TRAIN_DATA_PATH)
    if val_data:
        save_disease_data_to_txt(val_data, VAL_DATA_PATH)
    else:
        try:
            if os.path.exists(VAL_DATA_PATH):
                os.remove(VAL_DATA_PATH)
        except Exception:
            pass

    print(f"训练集: {len(train_data)}条疾病症状数据")
    print(f"验证集: {len(val_data)}条疾病症状数据" if val_data else "无验证集")

    if data:
        type_counts = {}
        for _, symptom_type in data:
            type_name = DISEASE_SYMPTOM_TYPES.get(symptom_type, f'未知类型{symptom_type}')
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        print("\n疾病类型分布:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}条")


if __name__ == "__main__":
    prepare_disease_data()