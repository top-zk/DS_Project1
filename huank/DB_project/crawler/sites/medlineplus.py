from __future__ import annotations
import logging
from typing import Iterable, List
from bs4 import BeautifulSoup
import string

from ..models import DiseaseRecord
from ..utils import polite_sleep, stable_weight_from_text, detect_urgency, classify_symptom_type


BASE = "https://medlineplus.gov"
INDEX = f"{BASE}/healthtopics.html"


def iter_condition_links(session) -> Iterable[str]:
    # MedlinePlus has an A-Z index.
    # The main page healthtopics.html usually links to pages like healthtopics_a.html, etc.
    # But checking the actual site structure via code logic:
    # We will try to find the alpha links first.
    
    try:
        r = session.get(INDEX)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        logging.error(f"Failed to fetch MedlinePlus index: {e}")
        return

    # Find A-Z links
    alpha_links = []
    # Usually they are in a navigation or specific container.
    # We'll look for links that look like healthtopics_[letter].html or similar.
    # Or just iterate 'A' through 'Z' if we can guess the URL pattern.
    # MedlinePlus pattern: https://medlineplus.gov/healthtopics_a.html ... _z.html
    # Let's generate them to be robust, or try to scrape them.
    
    # Try generating
    for char in string.ascii_lowercase:
        alpha_links.append(f"{BASE}/healthtopics_{char}.html")
    
    # Also add the main one just in case it lists 'A' or popular ones
    # alpha_links.append(INDEX) # INDEX is likely just a landing page, might redirect to A or list categories.
    
    for alpha_url in alpha_links:
        logging.info(f"Scanning index page: {alpha_url}")
        try:
            r = session.get(alpha_url)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "lxml")
            # MedlinePlus topic links are usually in <ul class="az-results"> or similar
            # Look for all links starting with /health/ or https://medlineplus.gov/health/
            # Exclude PDF or other languages if possible
            
            # The previous code used .az-results ul
            container = soup.select_one(".az-results") or soup
            
            found_count = 0
            for a in container.find_all("a", href=True):
                href = a.get("href")
                # Filter valid condition pages
                # Valid pages are usually like /ency/article/... or just /... .html
                # But previous code said /health/. Let's trust that but be broader if needed.
                # Actually MedlinePlus health topics are at https://medlineplus.gov/[topic].html or https://medlineplus.gov/health/[topic].html
                # Let's stick to the previous filter but ensure we catch full URLs
                
                if href:
                    href = href.strip()
                    # Filter valid condition pages
                    # Valid pages are usually like /ency/article/... or just /... .html
                    # But previous code said /health/. Let's trust that but be broader if needed.
                    # Actually MedlinePlus health topics are at https://medlineplus.gov/[topic].html or https://medlineplus.gov/health/[topic].html
                    # Let's stick to the previous filter but ensure we catch full URLs
                    
                    if href.startswith("https://medlineplus.gov/") or not href.startswith("http"):
                        full_url = href if href.startswith("http") else BASE + ("/" + href.lstrip("/") if not href.startswith("/") else href)
                    
                    # Heuristic to identify topic pages vs navigation
                    # Topic pages usually don't contain 'healthtopics' in filename (except the index ones)
                    if "healthtopics" in full_url:
                        continue
                    
                    # Common pattern for health topics
                    # https://medlineplus.gov/asthma.html
                    # https://medlineplus.gov/diabetes.html
                    # OR /genetics/condition/
                    
                    # Previous code used: href.startswith("/health/")
                    # Let's broaden to .html pages that are not index pages
                    if full_url.endswith(".html"):
                        # Exclude known non-disease pages
                        if any(x in full_url.lower() for x in ["encyclopedia", "druginformation", "directories", "organizations", "magazine", "videos", "recipe", "tutorial", "sitemap", "about"]):
                            continue
                            
                        yield full_url
                        found_count += 1
                        
            logging.info(f"Found {found_count} links on {alpha_url}")
            
        except Exception as e:
            logging.warning(f"Error scanning {alpha_url}: {e}")
            continue


def extract_section_text(soup, keywords: List[str]) -> str:
    text_content = []
    # MedlinePlus often uses h2 for sections
    for h in soup.find_all(["h2", "h3"]):
        if any(k in h.get_text(strip=True).lower() for k in keywords):
            curr = h.find_next_sibling()
            while curr and curr.name not in ["h1", "h2", "h3", "section"]:
                text_content.append(curr.get_text(" ", strip=True))
                curr = curr.find_next_sibling()
            break
    return " ".join(text_content).strip()

def parse_condition(session, url: str) -> DiseaseRecord | None:
    try:
        r = session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        
        # Check if valid page (has h1)
        h1 = soup.select_one("h1")
        if not h1:
            return None
            
        name = h1.get_text(strip=True)
        
        # Identify description
        # Added #topic-summary to the primary selector list
        desc_el = soup.select_one("#anch_0, .section-content, .summary-box, #topic-summary")
        desc = desc_el.get_text(" ", strip=True) if desc_el else ""
        if not desc:
            # Fallback to first few paragraphs
            ps = soup.select("article p")
            if ps:
                desc = " ".join(p.get_text(strip=True) for p in ps[:2])
        
        symptoms: List[str] = []
        for h in soup.find_all(["h2", "h3"]):
            if "symptom" in h.get_text(strip=True).lower():
                ul = h.find_next("ul")
                if ul:
                    symptoms = [li.get_text(strip=True) for li in ul.find_all("li")]
                break
        
        causes = extract_section_text(soup, ["causes", "etiology"])
        treatment = extract_section_text(soup, ["treatment", "therapy", "therapies"])
        prevention = extract_section_text(soup, ["prevention", "preventing"])

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
            病因=causes,
            治疗方法=treatment,
            预防措施=prevention
        )
        return record
    except Exception as e:
        logging.warning(f"MedlinePlus 解析失败: {url} err={e}")
        return None


def crawl(session, min_delay: float, max_delay: float, visited_manager=None) -> Iterable[DiseaseRecord]:
    # results: List[DiseaseRecord] = []
    for url in iter_condition_links(session):
        if visited_manager and url in visited_manager:
            continue
            
        rec = parse_condition(session, url)
        if rec:
            # results.append(rec)
            yield rec
            if visited_manager:
                visited_manager.add(url)
        polite_sleep(min_delay, max_delay)
    # return results
