import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse

def scrape_disease_data(url):
    """
    Scrapes disease data from a MedlinePlus page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract disease name (using the page title for now)
        disease_name = soup.find('h1').get_text(strip=True) if soup.find('h1') else "N/A"

        # Extract summary
        summary_div = soup.find('div', id='topic-summary')
        summary_text = ""
        if summary_div:
            # Find the first p tag in the summary div
            p_tag = summary_div.find('p')
            if p_tag:
                summary_text = p_tag.get_text(strip=True)

        # Extract symptoms (primary and possible secondary if multiple lists exist)
        symptoms = []
        secondary_symptoms = []
        if summary_div:
            symptoms_h3 = summary_div.find('h3', string='What are the symptoms of pneumonia?')
            if symptoms_h3:
                # First UL after the symptoms heading -> primary symptoms
                first_ul = symptoms_h3.find_next_sibling('ul')
                if first_ul:
                    symptoms = [li.get_text(strip=True) for li in first_ul.find_all('li')]
                # Any subsequent ULs before the next heading -> treat as secondary symptoms
                next_tag = first_ul.find_next_sibling() if first_ul else None
                while next_tag and next_tag.name != 'h3':
                    if next_tag.name == 'ul':
                        for li in next_tag.find_all('li'):
                            text = li.get_text(strip=True)
                            if text:
                                secondary_symptoms.append(text)
                    elif next_tag.name == 'p':
                        txt = next_tag.get_text(strip=True)
                        low = txt.lower()
                        # Heuristic extraction for secondary symptoms phrases in paragraphs
                        keyword_map = {
                            'lower-than-normal temperature': ['lower-than-normal temperature'],
                            'confusion': ['confused', 'confusion', 'delirium'],
                            'weakness': ['feel weak', 'weakness'],
                            'lack of energy': ['no energy', 'lack of energy'],
                            'restlessness': ['restless', 'restlessness']
                        }
                        for label, kws in keyword_map.items():
                            if any(k in low for k in kws):
                                if label not in secondary_symptoms and label not in symptoms:
                                    secondary_symptoms.append(label)
                    next_tag = next_tag.find_next_sibling()

        def extract_symptoms_from_url(page_url):
            try:
                r = requests.get(page_url, timeout=10)
                r.raise_for_status()
                sp = BeautifulSoup(r.content, 'html.parser')
                collected = []
                # Strategy 1: Summary box with an h3 like "What are the symptoms ..."
                summary = sp.find('div', id='topic-summary')
                if summary:
                    h3 = summary.find(lambda t: t and t.name == 'h3' and 'symptom' in t.get_text(strip=True).lower())
                    if h3:
                        ul = h3.find_next_sibling('ul')
                        if ul:
                            for li in ul.find_all('li'):
                                txt = li.get_text(strip=True)
                                if txt:
                                    collected.append(txt)
                            return collected

                # Strategy 2: Section heading H2/H3 that contains the word "Symptoms"
                heading = sp.find(lambda t: t and t.name in ['h2', 'h3', 'h4'] and 'symptom' in t.get_text(strip=True).lower())
                if heading:
                    tag = heading.find_next_sibling()
                    while tag and tag.name not in ['h2', 'h3']:
                        if tag.name in ['ul', 'ol']:
                            for li in tag.find_all('li'):
                                txt = li.get_text(strip=True)
                                if txt:
                                    collected.append(txt)
                        tag = tag.find_next_sibling()
                return collected
            except requests.exceptions.RequestException:
                return []

        # Crawl related MedlinePlus pages to enrich secondary symptoms
        related_urls = []
        def collect_urls_from_section(h2_title):
            h2 = soup.find(lambda t: t and t.name == 'h2' and t.get_text(strip=True) == h2_title)
            if not h2:
                return
            # Prefer container-based collection
            container = h2.find_parent(lambda t: t and t.has_attr('class'))
            nodes = []
            if container:
                nodes.append(container)
            # Include siblings until next H2 as fallback
            sib = h2.find_next_sibling()
            while sib and sib.name != 'h2':
                nodes.append(sib)
                sib = sib.find_next_sibling()
            for node in nodes:
                for a in node.find_all('a', href=True):
                    href = a['href']
                    abs_url = urljoin(url, href)
                    if urlparse(abs_url).netloc.endswith('medlineplus.gov'):
                        related_urls.append(abs_url)

        collect_urls_from_section('Related Health Topics')
        collect_urls_from_section('Specifics')
        # De-duplicate and limit to avoid over-fetching
        dedup = []
        seen_u = set()
        for u in related_urls:
            if u not in seen_u:
                seen_u.add(u)
                dedup.append(u)
        for u in dedup[:6]:
            for item in extract_symptoms_from_url(u):
                if item and item not in symptoms and item not in secondary_symptoms:
                    secondary_symptoms.append(item)

        # Extract examination indicators
        examination_indicators = []
        if summary_div:
            diagnosis_h3 = summary_div.find('h3', string='How is pneumonia diagnosed?')
            if diagnosis_h3:
                # Find the next ul tag
                diagnosis_ul = diagnosis_h3.find_next_sibling('ul')
                if diagnosis_ul:
                    for li in diagnosis_ul.find_all('li', recursive=False):
                        if li.find('ul'):
                            # Get the text of the li itself, without the nested ul
                            text = ''.join(li.find_all(string=True, recursive=False)).strip()
                            if text:
                                examination_indicators.append(text)
                            # Then get the text of the nested li's
                            for nested_li in li.find('ul').find_all('li'):
                                examination_indicators.append(nested_li.get_text(strip=True))
                        else:
                            examination_indicators.append(li.get_text(strip=True))

        # Extract differential diagnoses (using "Related Issues" section if present)
        differential_diagnoses = []
        # Prefer H2 sections: "Related Issues" and "Related Health Topics"
        related_h2 = soup.find(lambda tag: tag.name == 'h2' and tag.get_text(strip=True) == 'Related Issues')
        if related_h2:
            # Try to locate the enclosing section container and collect links inside
            section_container = related_h2.find_parent(lambda t: t.has_attr('class') and ('section' in t.get('class', []) or 'section-header' in t.get('class', []) or 'section-title' in t.get('class', [])))
            if section_container:
                for a in section_container.find_all('a'):
                    text = a.get_text(strip=True)
                    if text:
                        differential_diagnoses.append(text)
            else:
                # Fallback: traverse siblings
                tag = related_h2.find_next_sibling()
                while tag and not (tag.name == 'h2'):
                    for a in tag.find_all('a'):
                        text = a.get_text(strip=True)
                        if text:
                            differential_diagnoses.append(text)
                    tag = tag.find_next_sibling()

        topics_h2 = soup.find(lambda tag: tag.name == 'h2' and tag.get_text(strip=True) == 'Related Health Topics')
        if topics_h2:
            side_container = topics_h2.find_parent(lambda t: t.has_attr('class') and ('side-section' in t.get('class', []) or 'section' in t.get('class', [])))
            if side_container:
                for a in side_container.find_all('a'):
                    text = a.get_text(strip=True)
                    if text:
                        differential_diagnoses.append(text)
            else:
                tag = topics_h2.find_next_sibling()
                while tag and not (tag.name == 'h2'):
                    for a in tag.find_all('a'):
                        text = a.get_text(strip=True)
                        if text:
                            differential_diagnoses.append(text)
                    tag = tag.find_next_sibling()

        # De-duplicate while preserving order
        seen = set()
        differential_diagnoses = [x for x in differential_diagnoses if not (x in seen or seen.add(x))]


        # For demonstration, I'll create a dictionary with the extracted data
        disease_data = {
            "疾病名称": disease_name,
            "来源": url,
            "摘要": summary_text,
            "主要症状": symptoms,
            "次要症状": secondary_symptoms,
            "检查指标": examination_indicators,
            "诊断标准": "主要症状≥2项 + 检查指标异常", # Placeholder
            "鉴别诊断": differential_diagnoses
        }

        return disease_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

if __name__ == "__main__":
    pneumonia_url = "https://medlineplus.gov/pneumonia.html"
    scraped_data = scrape_disease_data(pneumonia_url)

    if scraped_data:
        # Save the data to a JSON file
        with open("pneumonia.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        print("Data scraped and saved to pneumonia.json")