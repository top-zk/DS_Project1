from __future__ import annotations
import logging
import json
from typing import List, Dict, Any

from .config import CrawlConfig
from .utils import setup_logger, build_session, polite_sleep, make_record_id
from .storage import JsonStorage, MongoStorage, MySQLStorage
from .models import DiseaseRecord
from .sites import medlineplus, nhsuk, icd10data
from .icd_lookup import lookup_icd10


def run() -> Dict[str, Any]:
    cfg = CrawlConfig()
    setup_logger(cfg.log_path)
    session = build_session(cfg.user_agent, cfg.timeout_sec, cfg.max_retries)

    logging.info("开始爬取 MedlinePlus")
    mp_records = medlineplus.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec)
    logging.info(f"MedlinePlus 获取 {len(mp_records)} 条")

    logging.info("开始爬取 NHS.uk")
    nhs_records = nhsuk.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec)
    logging.info(f"NHS.uk 获取 {len(nhs_records)} 条")

    logging.info("开始爬取 ICD10Data")
    icd_records = icd10data.crawl(session, cfg.min_delay_sec, cfg.max_delay_sec)
    logging.info(f"ICD10Data 获取 {len(icd_records)} 条")

    all_records: List[DiseaseRecord] = []
    # 先放入ICD10（编码完备）
    all_records.extend(icd_records)

    # 对其他站点，尽量补全ICD编码
    for rec in mp_records + nhs_records:
        if rec.disease_code == "UNKNOWN":
            code = lookup_icd10(session, rec.disease_name)
            rec.disease_code = code or "UNKNOWN"
        all_records.append(rec)
        polite_sleep(cfg.min_delay_sec, cfg.max_delay_sec)

    # 去重：按 name+source_url
    uniq: Dict[str, DiseaseRecord] = {}
    for r in all_records:
        rid = make_record_id(r.disease_name, r.source_url)
        if rid not in uniq:
            uniq[rid] = r

    records = list(uniq.values())
    logging.info(f"去重后记录数: {len(records)}")

    # 质量校验与清洗：丢弃空必填字段，修正概率范围
    cleaned: List[DiseaseRecord] = []
    bad: List[Dict[str, Any]] = []
    for r in records:
        try:
            r.probability_weight = max(0.0, min(1.0, r.probability_weight))
            # 重新构建模型以触发校验
            cleaned.append(DiseaseRecord(**r.to_chinese_json()))
        except Exception as e:
            bad.append({"name": getattr(r, "disease_name", ""), "url": getattr(r, "source_url", ""), "err": str(e)})

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
            "MedlinePlus": len(mp_records),
            "NHS.uk": len(nhs_records),
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

