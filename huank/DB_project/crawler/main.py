from __future__ import annotations
import logging
import json
from typing import List, Dict, Any
import os

from .config import CrawlConfig
from .utils import setup_logger, build_session, polite_sleep, make_record_id, VisitedManager
from .storage import JsonStorage, MongoStorage, MySQLStorage
from .models import DiseaseRecord
from .sites import medlineplus, nhsuk, icd10data
from .icd_lookup import lookup_icd10


def run() -> Dict[str, Any]:
    cfg = CrawlConfig()
    setup_logger(cfg.log_path)
    session = build_session(cfg.user_agent, cfg.timeout_sec, cfg.max_retries)
    
    # Initialize Visited Manager
    visited_mgr = VisitedManager(os.path.join(os.path.dirname(cfg.log_path), "visited_urls.txt"))
    logging.info(f"Loaded {len(visited_mgr.visited)} visited URLs.")

    # Load existing data to prevent overwriting
    all_records: List[DiseaseRecord] = []
    if os.path.exists(cfg.output_json_path):
        try:
            with open(cfg.output_json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    for item in existing_data:
                        try:
                            all_records.append(DiseaseRecord(**item))
                        except Exception as e:
                            logging.warning(f"Failed to load existing record: {e}")
            logging.info(f"Loaded {len(all_records)} existing records from {cfg.output_json_path}")
        except Exception as e:
            logging.error(f"Error loading existing data: {e}")

    # Helper for saving
    def save_batch(records_to_save):
        uniq: Dict[str, DiseaseRecord] = {}
        for r in records_to_save:
            rid = make_record_id(r.disease_name, r.source_url)
            # Always overwrite to keep the latest version
            uniq[rid] = r
        
        cleaned_list = []
        for r in uniq.values():
            try:
                 # Ensure valid
                 cleaned_list.append(DiseaseRecord(**r.to_chinese_json()))
            except Exception:
                 pass
        
        JsonStorage(cfg.output_json_path).write_all(cleaned_list)
        return cleaned_list

    logging.info("开始爬取 MedlinePlus")
    mp_count = 0
    mp_records = []
    for rec in medlineplus.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec, visited_mgr):
        mp_records.append(rec)
        all_records.append(rec)
        mp_count += 1
        logging.info(f"[Progress] MedlinePlus 抓取: {rec.disease_name}")
        if mp_count % 5 == 0:
            save_batch(all_records)
    logging.info(f"MedlinePlus 获取 {mp_count} 条")

    logging.info("开始爬取 NHS.uk")
    nhs_count = 0
    nhs_records = []
    for rec in nhsuk.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec, visited_mgr):
        nhs_records.append(rec)
        all_records.append(rec)
        nhs_count += 1
        logging.info(f"[Progress] NHS.uk 抓取: {rec.disease_name}")
        if nhs_count % 5 == 0:
            save_batch(all_records)
    logging.info(f"NHS.uk 获取 {nhs_count} 条")

    logging.info("开始爬取 ICD10Data")
    # ICD10Data returns list, keep as is
    icd_records = icd10data.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec)
    logging.info(f"ICD10Data 获取 {len(icd_records)} 条")
    all_records.extend(icd_records)

    # 对其他站点，尽量补全ICD编码
    # Note: Only for new records to save time, or all? Let's do all for now but might be slow.
    # To be efficient, we can check if code is UNKNOWN.
    for rec in all_records:
        if rec.disease_code == "UNKNOWN":
            # Only lookup if we have session and it's not ICD10 data
            # Skipping lookup for now to speed up the loop unless strictly needed
            # code = lookup_icd10(session, rec.disease_name)
            # rec.disease_code = code or "UNKNOWN"
            pass

    # Final Save and Stats
    # 去重：按 name+source_url
    uniq: Dict[str, DiseaseRecord] = {}
    for r in all_records:
        rid = make_record_id(r.disease_name, r.source_url)
        uniq[rid] = r

    records = list(uniq.values())
    logging.info(f"最终去重后记录数: {len(records)}")

    # 质量校验与清洗
    cleaned: List[DiseaseRecord] = []
    bad: List[Dict[str, Any]] = []
    for r in records:
        try:
            r.probability_weight = max(0.0, min(1.0, r.probability_weight))
            cleaned.append(DiseaseRecord(**r.to_chinese_json()))
        except Exception as e:
            bad.append({"name": getattr(r, "disease_name", ""), "url": getattr(r, "source_url", ""), "err": str(e)})
            logging.warning(f"Validation error for {getattr(r, 'disease_name', 'Unknown')}: {e}")

    logging.info(f"清洗后有效记录数: {len(cleaned)} 丢弃: {len(bad)}")

    # 写入JSON
    JsonStorage(cfg.output_json_path).write_all(cleaned)

    # 尝试写入Mongo与MySQL（若可连通）
    mongo_written = 0
    try:
        mstore = MongoStorage(cfg.mongo_url, cfg.mongo_db, cfg.mongo_collection)
        for r in cleaned:
            rid = make_record_id(r.disease_name, r.source_url)
            mstore.upsert(r, rid)
            mongo_written += 1
        logging.info(f"MongoDB 写入 {mongo_written} 条")
    except Exception as e:
        logging.warning(f"MongoDB 写入失败: {e}")

    mysql_written = 0
    try:
        msql = MySQLStorage(cfg.mysql_url)
        for r in cleaned:
            rid = make_record_id(r.disease_name, r.source_url)
            msql.upsert(r, rid)
            mysql_written += 1
        logging.info(f"MySQL 写入 {mysql_written} 条")
    except Exception as e:
        logging.warning(f"MySQL 写入失败: {e}")

    # 统计报告
    stats = {
        "total": len(cleaned),
        "sources": {
            "MedlinePlus": mp_count,
            "NHS.uk": nhs_count,
            "ICD10Data": len(icd_records),
        },
        "mongo_written": mongo_written,
        "mysql_written": mysql_written,
        "dropped": len(bad),
        "dropped_samples": bad[:10],
    }
    with open(CrawlConfig().stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logging.info(f"统计报告写入: {CrawlConfig().stats_path}")
    return stats


if __name__ == "__main__":
    run()
