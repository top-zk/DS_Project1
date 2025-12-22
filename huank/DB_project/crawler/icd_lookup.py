from __future__ import annotations
import logging
from typing import Optional
from bs4 import BeautifulSoup


SEARCH = "https://www.icd10data.com/search?s={q}"


def lookup_icd10(session, disease_name: str) -> Optional[str]:
    try:
        r = session.get(SEARCH.format(q=disease_name))
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        a = soup.select_one(".search-results a")
        if not a:
            return None
        code = a.get_text(strip=True)
        if code and len(code) <= 10:
            return code
        return None
    except Exception as e:
        logging.debug(f"ICD10 搜索失败: {disease_name} err={e}")
        return None

