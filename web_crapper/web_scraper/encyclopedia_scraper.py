import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

BASE_URL = "https://medlineplus.gov/ency/encyclopedia_A.htm"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encyclopedia_data")

def get_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_letter_urls():
    base_url = "https://medlineplus.gov/ency/encyclopedia_{}.htm"
    return [base_url.format(chr(ord('A') + i)) for i in range(26)]

def get_topic_urls(letter_url):
    soup = get_soup(letter_url)
    if not soup:
        return []

    topic_urls = []
    for link in soup.select("ul#index > li > a"):
        topic_urls.append(urljoin(letter_url, link['href']))
    return topic_urls

def scrape_topic_page(topic_url):
    soup = get_soup(topic_url)
    if not soup:
        return None

    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else ""
    summary_div = soup.find('div', id='topic-summary')
    summary = summary_div.get_text(strip=True) if summary_div else ""

    return {
        "url": topic_url,
        "title": title,
        "summary": summary
    }

def sanitize_filename(name):
    """ Sanitize a string to be used as a filename. """
    name = re.sub(r'[\\/*?:"<>|]',"", name)
    name = name.replace(' ', '_')
    return name

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Starting scraper...")
    letter_urls = get_letter_urls()
    print(f"Found {len(letter_urls)} letter pages to scrape.")

    for letter_url in letter_urls:
        print(f"Scraping topics from {letter_url}")
        topic_urls = get_topic_urls(letter_url)
        print(f"Found {len(topic_urls)} topics.")

        for topic_url in topic_urls:
            print(f"Scraping {topic_url}")
            topic_data = scrape_topic_page(topic_url)
            if topic_data and topic_data['title']:
                try:
                    filename = sanitize_filename(topic_data['title'])
                    if not filename:
                        print(f"Skipping topic with empty sanitized title: {topic_url}")
                        continue
                    
                    filepath = os.path.join(DATA_DIR, f"{filename}.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(topic_data, f, ensure_ascii=False, indent=4)
                    print(f"Saved data to {filepath}")
                except Exception as e:
                    print(f"Could not save file for title: {topic_data['title']}. Error: {e}")
            elif topic_data:
                print(f"Skipping topic with no title: {topic_url}")
            
            time.sleep(0.1)

        time.sleep(1)

if __name__ == "__main__":
    main()