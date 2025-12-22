from __future__ import annotations
import logging
from typing import Iterable, List
from bs4 import BeautifulSoup

from ..models import DiseaseRecord
from ..utils import polite_sleep, stable_weight_from_text, detect_urgency, classify_symptom_type


BASE = "https://www.icd10data.com"
INDEX = f"{BASE}/ICD10CM/Codes"


def iter_code_links(session) -> Iterable[str]:
    r = session.get(INDEX)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    links = []
    for a in soup.select(".code-list a"):
        href = a.get("href")
        if href and href.startswith("/ICD10CM/Codes"):
            links.append(BASE + href)
    logging.info(f"ICD10Data 类别链接数={len(links)}")
    # 深入到更多层级
    for url in links:
        rr = session.get(url)
        rr.raise_for_status()
        s2 = BeautifulSoup(rr.text, "lxml")
        for a in s2.select(".code-list a"):
            href = a.get("href")
            if href and href.startswith("/ICD10CM/Codes"):
                yield BASE + href


def parse_code_page(session, url: str) -> List[DiseaseRecord]:
    try:
        r = session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        records: List[DiseaseRecord] = []
        for row in soup.select(".codeTable tr"):
            code_el = row.select_one("td.code a")
            title_el = row.select_one("td.desc a, td.desc")
            if not code_el or not title_el:
                continue
            code = code_el.get_text(strip=True)
            name = title_el.get_text(strip=True)
            desc = soup.select_one(".lead, .content")
            desc_txt = desc.get_text(" ", strip=True) if desc else name
            symptom_type = classify_symptom_type(desc_txt + name)
            urgency = detect_urgency(desc_txt)
            weight = stable_weight_from_text(desc_txt + name)
            rec = DiseaseRecord(
                疾病名称=name,
                疾病编码=code,
                主要症状=[],
                症状描述=desc_txt,
                症状类型=symptom_type,
                概率权重=weight,
                紧急程度=urgency,
                来源链接=url,
            )
            records.append(rec)
        return records
    except Exception as e:
        logging.warning(f"ICD10Data 解析失败: {url} err={e}")
        return []


def crawl(session, min_delay: float, max_delay: float) -> List[DiseaseRecord]:
    all_records: List[DiseaseRecord] = []
    for url in iter_code_links(session):
        recs = parse_code_page(session, url)
        all_records.extend(recs)
        polite_sleep(min_delay, max_delay)
    return all_records

