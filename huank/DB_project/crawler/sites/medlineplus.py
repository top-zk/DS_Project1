from __future__ import annotations
import logging
from typing import Iterable, List
from bs4 import BeautifulSoup

from ..models import DiseaseRecord
from ..utils import polite_sleep, stable_weight_from_text, detect_urgency, classify_symptom_type


BASE = "https://medlineplus.gov"
INDEX = f"{BASE}/healthtopics.html"


def iter_condition_links(session) -> Iterable[str]:
    r = session.get(INDEX)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    links = []
    for ul in soup.select(".az-results ul"):  # A-Z list blocks
        for a in ul.select("li a"):
            href = a.get("href")
            if href and href.startswith("/health/"):
                links.append(BASE + href)
    logging.info(f"MedlinePlus 条目链接数={len(links)}")
    for url in links:
        yield url


def parse_condition(session, url: str) -> DiseaseRecord | None:
    try:
        r = session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        name = soup.select_one("h1").get_text(strip=True)
        desc_el = soup.select_one("#anch_0, .section-content")
        desc = desc_el.get_text(" ", strip=True) if desc_el else soup.get_text(" ", strip=True)
        symptoms: List[str] = []
        for h in soup.find_all(["h2", "h3"]):
            if "symptom" in h.get_text(strip=True).lower():
                ul = h.find_next("ul")
                if ul:
                    symptoms = [li.get_text(strip=True) for li in ul.find_all("li")]
                break
        symptom_type = classify_symptom_type(desc)
        urgency = detect_urgency(desc)
        weight = stable_weight_from_text(desc + name)
        # 暂缺编码：后续通过ICD10检索填充
        code = ""  # 先留空，由主流程补全
        record = DiseaseRecord(
            疾病名称=name,
            疾病编码=code or "UNKNOWN",
            主要症状=symptoms,
            症状描述=desc,
            症状类型=symptom_type,
            概率权重=weight,
            紧急程度=urgency,
            来源链接=url,
        )
        return record
    except Exception as e:
        logging.warning(f"MedlinePlus 解析失败: {url} err={e}")
        return None


def crawl(session, min_delay: float, max_delay: float) -> List[DiseaseRecord]:
    results: List[DiseaseRecord] = []
    for url in iter_condition_links(session):
        rec = parse_condition(session, url)
        if rec:
            results.append(rec)
        polite_sleep(min_delay, max_delay)
    return results

