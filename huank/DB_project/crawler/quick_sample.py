from __future__ import annotations
import json
from typing import List, Dict, Any

from .config import CrawlConfig
from .utils import setup_logger, build_session, polite_sleep, make_record_id
from .storage import JsonStorage
from .models import DiseaseRecord
from .sites import nhsuk
from .icd_lookup import lookup_icd10


def run(limit: int = 50) -> Dict[str, Any]:
    cfg = CrawlConfig()
    setup_logger(cfg.log_path)
    session = build_session(cfg.user_agent, cfg.timeout_sec, cfg.max_retries)
    records: List[DiseaseRecord] = []
    count = 0
    for url in nhsuk.iter_condition_links(session):
        if count >= limit:
            break
        rec = nhsuk.parse_condition(session, url)
        if rec:
            if rec.disease_code == "UNKNOWN":
                code = lookup_icd10(session, rec.disease_name)
                rec.disease_code = code or "UNKNOWN"
            records.append(rec)
            count += 1
        polite_sleep(0.2, 0.4)

    uniq: Dict[str, DiseaseRecord] = {}
    for r in records:
        rid = make_record_id(r.disease_name, r.source_url)
        if rid not in uniq:
            uniq[rid] = r
    cleaned: List[DiseaseRecord] = []
    for r in uniq.values():
        r.probability_weight = max(0.0, min(1.0, r.probability_weight))
        cleaned.append(DiseaseRecord(**r.to_chinese_json()))

    JsonStorage(cfg.output_json_path).write_all(cleaned)
    return {"total": len(cleaned), "output": cfg.output_json_path}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))

