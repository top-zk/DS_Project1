import os
import re
import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Generator, Optional, Union, Iterator
from medical_config import DATA_PATH, SAMPLE_DATA, TRAIN_DATA_PATH, VAL_DATA_PATH, DISEASE_SYMPTOM_TYPES, DATA_DIR, ENCYCLOPEDIA_DIR, ENCYCLOPEDIA_JSON_DIR, DISEASE_JSON_PATH, DISEASE_XML_PATH, DEMO_JSON_PATH
from sklearn.model_selection import train_test_split
import subprocess
import importlib
import random


def install_required_packages():
    """安装必要的Python包"""
    required = ['xlrd', 'openpyxl']
    for package in required:
        try:
            importlib.import_module(package)
            # print(f"{package} 已安装")
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call(['pip', 'install', package])
            print(f"{package} 安装完成")


class DataLoader:
    """
    Standardized data loader for medical data.
    Supports JSON and CSV formats with validation, cleaning, and normalization.
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize the DataLoader.
        
        Args:
            batch_size (int): Number of items to yield per batch.
        """
        self.batch_size = batch_size
        
    def load(self, file_path: str) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Load data from a file (JSON or CSV) and yield batches of processed records.
        
        Args:
            file_path (str): Path to the data file.
            
        Yields:
            List[Dict[str, Any]]: A batch of processed data records.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.json':
            iterator = self._parse_json(file_path)
        elif ext == '.csv':
            iterator = self._parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        batch = []
        for item in iterator:
            processed = self._process_item(item)
            if processed:
                batch.append(processed)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch

    def _parse_json(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Parse JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Optimized for list of objects
                # Note: For very large files, consider line-delimited JSON or ijson
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                elif isinstance(data, dict):
                    # Handle specific wrapper formats like {'data': [...]}
                    if 'data' in data and isinstance(data['data'], list):
                        for item in data['data']:
                            yield item
                    else:
                        yield data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON {file_path}: {e}")

    def _parse_csv(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Parse CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
        except Exception as e:
            print(f"Error parsing CSV {file_path}: {e}")

    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate, clean, and normalize a single data item."""
        # 1. Normalize Keys
        normalized = {}
        
        # Mappings
        key_map = {
            '疾病名称': 'title', 'title': 'title', 'name': 'title',
            '主要症状': 'symptoms', 'symptoms': 'symptoms',
            '症状描述': 'content', 'content': 'content', 'summary': 'content', 'desc': 'content',
            '症状类型': 'category', 'category': 'category', 'label': 'category', 'department': 'category'
        }
        
        for k, v in item.items():
            if k in key_map:
                normalized[key_map[k]] = v
            else:
                normalized[k] = v # Keep other fields
        
        # 2. Validation
        if not normalized.get('title'):
            return None # Skip items without title
            
        # 3. Cleaning & Formatting
        if 'symptoms' in normalized:
            if isinstance(normalized['symptoms'], str):
                # Try to split by comma if it's a string
                normalized['symptoms'] = [s.strip() for s in re.split(r'[,，、]', normalized['symptoms']) if s.strip()]
            elif isinstance(normalized['symptoms'], list):
                normalized['symptoms'] = [str(s).strip() for s in normalized['symptoms'] if str(s).strip()]
        else:
            normalized['symptoms'] = []
            
        if 'content' in normalized:
            normalized['content'] = clean_text(str(normalized['content']))
        else:
            normalized['content'] = ""
            
        # 4. Auto-Categorization (Optional integration with existing logic)
        if 'category' in normalized and normalized['category']:
             catn = normalize_category(normalized['category'])
             if catn:
                 normalized['category'] = catn
        
        return normalized


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

def is_chinese(string):
    """Check if the string contains Chinese characters"""
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def clean_text(t):
    if not t:
        return t
    s = str(t)
    # Remove URLs
    s = re.sub(r"http[s]?://\S+", "", s)
    # Remove metadata prefixes
    s = re.sub(r"(来源|网页源|Source|URL|链接)[:：].*", "", s)
    s = re.sub(r"(爬取时间|采集时间|抓取时间|时间|Time)[:：].*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    
    # Filter out if it doesn't contain any Chinese characters (likely English garbage)
    if not is_chinese(s):
        return ""
        
    return s

def auto_label_by_keywords(text):
    """根据关键词自动标注"""
    from medical_config import DISEASE_SYMPTOM_KEYWORDS
    
    max_count = 0
    best_label = None
    
    for label, keywords in DISEASE_SYMPTOM_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if kw in text:
                count += 1
        if count > max_count:
            max_count = count
            best_label = label
            
    return best_label if max_count > 0 else None

def load_disease_data_from_json(file_path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        items = obj if isinstance(obj, list) else [obj] # Handle single object
        
        name_to_id = {v: k for k, v in DISEASE_SYMPTOM_TYPES.items()}
        
        for it in items:
            # Try to get text from various fields
            text_parts = []
            if it.get('疾病名称'): text_parts.append(str(it.get('疾病名称')))
            if it.get('主要症状'): 
                symptoms = it.get('主要症状')
                if isinstance(symptoms, list):
                    text_parts.extend([str(s) for s in symptoms])
                else:
                    text_parts.append(str(symptoms))
            
            desc = it.get('症状描述') or it.get('text') or it.get('symptom') or it.get('content') or it.get('summary') or ''
            if desc: text_parts.append(str(desc))
            
            text = " ".join(text_parts).strip()
            text = clean_text(text)
            
            if not text:
                continue

            # Try to get label
            label = None
            label_val = it.get('症状类型') or it.get('label') or it.get('category')
            
            if label_val:
                catn = normalize_category(label_val)
                if catn:
                    label = name_to_id.get(catn)
            
            # If no label, try auto-labeling
            if label is None:
                label = auto_label_by_keywords(text)
                
            if label is not None:
                 result.append((text, int(label)))
                 
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


def load_disease_data_from_db(db_path='instance/medical_app.db'):
    """从数据库加载疾病数据"""
    import sqlite3
    if not os.path.exists(db_path):
        if not os.path.exists(os.path.abspath(db_path)):
             return []
    
    try:
        conn = sqlite3.connect(db_path)
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='encyclopedia';")
        if not cursor.fetchone():
            conn.close()
            return []

        query = "SELECT content, department FROM encyclopedia"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        data = []
        name_to_id = {v: k for k, v in DISEASE_SYMPTOM_TYPES.items()}
        
        for _, row in df.iterrows():
            text = clean_text(str(row['content']))
            dept = row['department']
            
            label = name_to_id.get(dept, 7) # default to 7 if not found
            
            if text:
                data.append((text, label))
                
        print(f"从数据库加载了 {len(data)} 条数据")
        return data
    except Exception as e:
        print(f"从数据库加载失败: {e}")
        return []

def clean_and_validate_data(data):
    """数据清洗和验证"""
    unique_data = {} 
    
    for text, label in data:
        if not text or len(text) < 5: 
            continue
        
        # Additional cleaning
        text = clean_text(text)
        
        if text not in unique_data:
            unique_data[text] = label
            
    validated_data = [(k, v) for k, v in unique_data.items()]
    print(f"清洗后剩余 {len(validated_data)} 条有效数据 (原始 {len(data)} 条)")
    return validated_data


def generate_synthetic_data_from_keywords():
    """Generate synthetic data from keywords"""
    from medical_config import DISEASE_SYMPTOM_KEYWORDS
    
    templates = [
        "我最近总是{0}",
        "感觉{0}",
        "{0}怎么办",
        "有没有治疗{0}的方法",
        "请问{0}是什么病",
        "最近{0}很难受",
        "医生，我{0}",
        "{0}好几天了",
        "出现{0}症状",
        "{0}，请问挂什么科",
        "总是{0}",
        "有点{0}",
        "一直{0}",
        "{0}很严重",
        "特别是{0}",
        "伴有{0}"
    ]
    
    synthetic_data = []
    for label, keywords in DISEASE_SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            # Add the keyword itself
            synthetic_data.append((kw, label))
            # Add templated versions
            for t in templates:
                synthetic_data.append((t.format(kw), label))
                
            # Combinations (pairs)
            if len(keywords) > 1:
                other_kw = random.choice(keywords)
                if other_kw != kw:
                    synthetic_data.append((f"{kw}和{other_kw}", label))
                    synthetic_data.append((f"既有{kw}又有{other_kw}", label))
                    
    print(f"生成的合成数据: {len(synthetic_data)} 条")
    return synthetic_data

def augment_data(data):
    """Augment data with templates"""
    templates = [
        "我最近总是{0}",
        "感觉{0}",
        "{0}怎么办",
        "有没有治疗{0}的方法",
        "请问{0}是什么病",
        "最近{0}很难受",
        "医生，我{0}",
        "{0}好几天了",
        "出现{0}症状",
        "{0}，请问挂什么科",
        "总是{0}",
        "有点{0}"
    ]
    
    augmented = []
    for text, label in data:
        augmented.append((text, label)) # Keep original
        
        # If text is short (likely a symptom name), apply templates
        if len(text) < 10:
             for t in templates:
                 augmented.append((t.format(text), label))
        
        # Also split comma separated symptoms
        parts = re.split(r'[，,、\s]+', text)
        if len(parts) > 1:
            for p in parts:
                p = p.strip()
                if len(p) > 1:
                     augmented.append((p, label))
                     for t in templates:
                        augmented.append((t.format(p), label))

    # Deduplicate
    return list(set(augmented))

def prepare_disease_data():
    """准备疾病数据并保存为文本文件"""
    install_required_packages()
    os.makedirs(DATA_DIR, exist_ok=True)

    data = []
    
    # 0) Load Demo JSON (High Quality Chinese Data)
    if os.path.exists(DEMO_JSON_PATH):
        print(f"检测到示例数据源: {DEMO_JSON_PATH}")
        demo_data = load_disease_data_from_json(DEMO_JSON_PATH)
        if demo_data:
            data.extend(demo_data)

    # 1) 优先尝试从 JSON 更新并重写 XML，确保数据最新
    if os.path.exists(DISEASE_JSON_PATH):
        print(f"检测到 JSON 数据源: {DISEASE_JSON_PATH}，正在重新生成 XML...")
        # Use new generic loader
        disease_json_pairs = load_disease_data_from_json(DISEASE_JSON_PATH)
        if disease_json_pairs:
            data.extend(disease_json_pairs)
    
    # 2) 如果没有 JSON，再尝试读取现有的 XML
    if not data and os.path.exists(DISEASE_XML_PATH):
        data = load_disease_data_from_xml(DISEASE_XML_PATH)

    # 3) 总是尝试加载百科目录
    json_data = []
    if os.path.isdir(ENCYCLOPEDIA_JSON_DIR):
        print(f"正在扫描百科目录: {ENCYCLOPEDIA_JSON_DIR}")
        for r, _, files in os.walk(ENCYCLOPEDIA_JSON_DIR):
            for n in files:
                if n.lower().endswith('.json'):
                    json_data.extend(load_disease_data_from_json(os.path.join(r, n)))
    if json_data:
        data.extend(json_data)
        
    # 4) 尝试从 TXT 目录加载
    txt_data = []
    if os.path.isdir(ENCYCLOPEDIA_DIR):
         txt_data = load_disease_data_from_directory(ENCYCLOPEDIA_DIR)
    if txt_data:
        data.extend(txt_data)

    if not data and os.path.exists(DATA_PATH):
        print(f"尝试从 DATA_PATH 加载: {DATA_PATH}")
        if DATA_PATH.lower().endswith('.json'):
             json_data = load_disease_data_from_json(DATA_PATH)
             if json_data:
                 data.extend(json_data)
        elif DATA_PATH.lower().endswith(('.xlsx', '.xls')):
            disease_data = load_disease_data_from_excel(DATA_PATH)
            data = [(item['症状描述'], item['症状类型']) for item in disease_data]
    if not data:
        data = [(item['症状描述'], item['症状类型']) for item in SAMPLE_DATA]

    if not data:
        data = [(item['症状描述'], item['症状类型']) for item in SAMPLE_DATA]
    
    # 尝试从数据库加载更多数据
    db_data = load_disease_data_from_db()
    if db_data:
        data.extend(db_data)

    # 数据清洗
    data = clean_and_validate_data(data)

    if not data:
        data = [(item['症状描述'], item['症状类型']) for item in SAMPLE_DATA]
    
    # 5) Generate Synthetic Data from Keywords (CRITICAL for coverage)
    syn_data = generate_synthetic_data_from_keywords()
    data.extend(syn_data)

    # Augment data with templates to increase size and robustness
    print(f"增强前数据量: {len(data)}")
    data = augment_data(data)
    print(f"增强后数据量: {len(data)}")

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
    # Test DataLoader
    print("Testing DataLoader...")
    loader = DataLoader(batch_size=2)
    test_json = DEMO_JSON_PATH if os.path.exists(DEMO_JSON_PATH) else None
    
    if test_json:
        print(f"Loading from {test_json}")
        for batch in loader.load(test_json):
            print(f"Loaded batch of size {len(batch)}")
            for item in batch:
                print(f"  - {item.get('title', 'No Title')}")
            break # Just one batch
    else:
        print("Demo JSON not found for testing.")
        
    prepare_disease_data()
