from __future__ import annotations
import logging
from typing import Iterable, List
from bs4 import BeautifulSoup
import string

from ..models import DiseaseRecord
from ..utils import polite_sleep, stable_weight_from_text, detect_urgency, classify_symptom_type


BASE = "https://www.nhs.uk"
INDEX = f"{BASE}/conditions/"


def iter_condition_links(session) -> Iterable[str]:
    # NHS structure: /conditions/ page lists A-Z links or all conditions.
    # Usually it has an A-Z navigation.
    # We will try to traverse A-Z if found, or assume the main page lists all (or categories).
    
    # Try to fetch the index
    try:
        r = session.get(INDEX)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        logging.error(f"Failed to fetch NHS index: {e}")
        return

    # Look for A-Z links. NHS typically has a list like 'A', 'B', 'C'...
    # The URL pattern might be /conditions/a/, /conditions/b/ etc? Or anchors?
    # Actually, NHS website often lists everything on /conditions/ or uses a load-more / A-Z filter.
    
    # Strategy:
    # 1. Scrape all links on the main /conditions/ page that point to specific conditions.
    # 2. Check if there are sub-pages for letters.
    
    # Let's try to generate letter URLs if they exist, or just rely on the main page having a list.
    # A common pattern is https://www.nhs.uk/conditions/a/ doesn't exist, it's just one big list or sections.
    # But let's check for links on the page that look like they belong to conditions.
    
    # Also, some sites use pagination.
    # If the page has "Next", we should follow it.
    
    # Simple robust approach: 
    # Get all links starting with /conditions/ that are not the index itself.
    
    seen_links = set()
    
    # Helper to process a page and yield links
    def process_page(soup):
        count = 0
        # NHS usually lists conditions in a list
        # Look for <a class="nhsuk-list-panel__link" ...> or just generic links
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            
            # Normalize
            if href.startswith("/"):
                full = BASE + href
            elif href.startswith("http"):
                full = href
            else:
                full = BASE + "/conditions/" + href
            
            # Filter
            # Must be under /conditions/
            if "/conditions/" not in full:
                continue
                
            # Avoid the index page itself
            if full.rstrip("/") == INDEX.rstrip("/"):
                continue
                
            # Avoid anchors
            if "#" in full:
                continue
                
            # Avoid obviously non-condition pages
            if full.endswith("index.aspx") or any(x in full.lower() for x in ["social-care", "medicines", "vaccinations", "pregnancy", "service-search"]):
                continue

            if full not in seen_links:
                seen_links.add(full)
                yield full
                count += 1
        return count

    # 1. Process main index
    yield from process_page(soup)
    
    # 2. Heuristic: Check for A-Z pages if we didn't find many
    # Maybe /conditions/ is just 'A' or 'Featured'.
    # Try fetching /conditions/b/ etc.
    for char in string.ascii_lowercase:
        alpha_url = f"{BASE}/conditions/{char}/" # Guessing pattern
        # If this 404s, we stop.
        # But actually NHS might not use this.
        # Let's try it for a few common letters.
        
        # Actually, let's look for "A-Z" links on the main page.
        # If there are links with text "A", "B", ... follow them.
        pass # Skipping blind guessing to avoid spamming 404s unless we are sure.

    # 3. Check for "Next" button
    # NHS pagination usually uses rel="next" or class="pagination__next"
    next_link = soup.find("a", attrs={"rel": "next"}) or soup.find("a", string=lambda t: t and "Next" in t)
    while next_link:
        href = next_link.get("href")
        if not href:
            break
        next_url = BASE + href if href.startswith("/") else href
        logging.info(f"Following Next Page: {next_url}")
        
        try:
            r = session.get(next_url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            yield from process_page(soup)
            
            next_link = soup.find("a", attrs={"rel": "next"}) or soup.find("a", string=lambda t: t and "Next" in t)
        except Exception as e:
            logging.error(f"Error following next page {next_url}: {e}")
            break


def extract_section_text(soup, keywords: List[str]) -> str:
    text_content = []
    for h in soup.find_all(["h2", "h3"]):
        if any(k in h.get_text(strip=True).lower() for k in keywords):
            curr = h.find_next_sibling()
            while curr and curr.name not in ["h1", "h2", "h3", "section", "article"]:
                text_content.append(curr.get_text(" ", strip=True))
                curr = curr.find_next_sibling()
            break
    return " ".join(text_content).strip()

def parse_condition(session, url: str) -> DiseaseRecord | None:
    try:
        r = session.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        
        h1 = soup.select_one("h1")
        if not h1:
            return None
        name = h1.get_text(strip=True)
        
        desc_el = soup.select_one("article")
        desc = desc_el.get_text(" ", strip=True) if desc_el else soup.get_text(" ", strip=True)
        
        symptoms: List[str] = []
        for h in soup.find_all(["h2", "h3"]):
            if "symptoms" in h.get_text(strip=True).lower():
                ul = h.find_next("ul")
                if ul:
                    symptoms = [li.get_text(strip=True) for li in ul.find_all("li")]
                break
        
        causes = extract_section_text(soup, ["causes", "why", "causes of"])
        treatment = extract_section_text(soup, ["treatment", "treating", "how to treat"])
        prevention = extract_section_text(soup, ["prevention", "preventing", "how to prevent"])

        symptom_type = classify_symptom_type(desc)
        urgency = detect_urgency(desc)
        weight = stable_weight_from_text(desc + name)
        record = DiseaseRecord(
            疾病名称=name,
            疾病编码="UNKNOWN",
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
        logging.warning(f"NHS.uk 解析失败: {url} err={e}")
        return None


def crawl(session, min_delay: float, max_delay: float, visited_manager=None) -> Iterable[DiseaseRecord]:
    # results: List[DiseaseRecord] = []
    for url in iter_condition_links(session):
        if visited_manager and url in visited_manager:
            continue
            
        rec = parse_condition(session, url)
        if rec:
            yield rec
            if visited_manager:
                visited_manager.add(url)
        polite_sleep(min_delay, max_delay)
    # return results
