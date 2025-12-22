from __future__ import annotations
import json
import logging
from typing import Iterable, Optional
from datetime import datetime

from .models import DiseaseRecord


class JsonStorage:
    def __init__(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write_all(self, records: Iterable[DiseaseRecord]) -> None:
        data = [r.to_chinese_json() for r in records]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"写入JSON: {self.path}, 记录数={len(data)}")


class MongoStorage:
    def __init__(self, url: str, db: str, collection: str):
        from pymongo import MongoClient
        from pymongo.errors import DuplicateKeyError
        self.client = MongoClient(url)
        self.collection = self.client[db][collection]
        self.collection.create_index("_id", unique=True)
        self.DuplicateKeyError = DuplicateKeyError

    def upsert(self, record: DiseaseRecord, _id: str) -> None:
        doc = record.to_chinese_json()
        doc["_id"] = _id
        self.collection.replace_one({"_id": _id}, doc, upsert=True)


class MySQLStorage:
    def __init__(self, url: str):
        from sqlalchemy import create_engine, text
        self.engine = create_engine(url, pool_pre_ping=True)
        with self.engine.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS diseases (
                    id VARCHAR(64) PRIMARY KEY,
                    疾病名称 VARCHAR(255) NOT NULL,
                    疾病编码 VARCHAR(64) NOT NULL,
                    主要症状 JSON,
                    症状描述 TEXT NOT NULL,
                    症状类型 VARCHAR(64) NOT NULL,
                    概率权重 DOUBLE NOT NULL,
                    紧急程度 VARCHAR(32) NOT NULL,
                    来源链接 TEXT NOT NULL,
                    爬取时间戳 DATETIME NOT NULL
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
                """
            ))

    def upsert(self, record: DiseaseRecord, _id: str) -> None:
        from sqlalchemy import text
        with self.engine.begin() as conn:
            conn.execute(text(
                """
                REPLACE INTO diseases (
                    id, 疾病名称, 疾病编码, 主要症状, 症状描述,
                    症状类型, 概率权重, 紧急程度, 来源链接, 爬取时间戳
                ) VALUES (:id, :name, :code, :symptoms, :desc,
                    :stype, :p, :urg, :url, :ts)
                """
            ), {
                "id": _id,
                "name": record.disease_name,
                "code": record.disease_code,
                "symptoms": json.dumps(record.main_symptoms, ensure_ascii=False),
                "desc": record.symptom_description,
                "stype": record.symptom_type,
                "p": record.probability_weight,
                "urg": record.urgency_level,
                "url": record.source_url,
                "ts": record.crawl_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            })

