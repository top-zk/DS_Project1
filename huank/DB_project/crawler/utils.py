import random
import time
import logging
from typing import Optional
import hashlib
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_logger(log_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def build_session(user_agent: str, timeout: float, max_retries: int) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.request = _wrap_request(session.request, timeout)
    return session


def _wrap_request(orig_request, timeout):
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return orig_request(method, url, **kwargs)
    return wrapped


def polite_sleep(min_delay: float, max_delay: float) -> None:
    time.sleep(random.uniform(min_delay, max_delay))


def stable_weight_from_text(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    return round(0.6 + (n % 350) / 1000.0, 3)


def detect_urgency(text: str) -> str:
    t = text.lower()
    if re.search(r"(seek urgent|emergency|call 999|call 911|急诊|危急)", t):
        return "高"
    if re.search(r"(see a doctor|medical advice|尽快就医|及时就医)", t):
        return "中等"
    return "低"


def classify_symptom_type(text: str) -> str:
    t = text.lower()
    mapping = {
        "呼吸系统": ["cough", "breath", "lung", "wheeze", "咳", "呼吸", "肺"],
        "消化系统": ["abdom", "nausea", "vomit", "diarr", "腹", "恶心", "腹泻"],
        "神经系统": ["headache", "seizure", "dizziness", "migraine", "头痛", "癫痫", "眩晕"],
        "心血管系统": ["chest pain", "palpit", "heart", "心", "胸痛", "心悸"],
        "肌肉骨骼": ["joint", "muscle", "骨", "肌肉", "关节"],
        "皮肤": ["rash", "itch", "皮疹", "瘙痒", "皮肤"],
        "泌尿生殖": ["urine", "urinary", "尿", "泌尿", "生殖"],
    }
    for k, kws in mapping.items():
        if any(kw in t for kw in kws):
            return k
    return "其他"


def make_record_id(name: str, source_url: str) -> str:
    return hashlib.sha256(f"{name}|{source_url}".encode("utf-8")).hexdigest()

