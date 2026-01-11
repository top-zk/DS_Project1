
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)

def check_url(url, name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        logging.info(f"Fetching {name}: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        logging.info(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check Title
            title = soup.find('h1')
            logging.info(f"Title found: {title.get_text().strip() if title else 'NO H1 FOUND'}")
            
            # Check content size
            text = soup.get_text()
            logging.info(f"Total text length: {len(text)}")
            
            # Check for specific container usually holding content
            # MedlinePlus: .main-content or #topic-summary
            # NHS: article
            
            if "medlineplus" in url:
                summary = soup.select_one("#topic-summary")
                logging.info(f"MedlinePlus Summary found: {bool(summary)}")
                if summary:
                    logging.info(f"Summary text: {summary.get_text()[:100]}...")
            elif "nhs.uk" in url:
                article = soup.select_one("article")
                logging.info(f"NHS Article found: {bool(article)}")
                if article:
                    logging.info(f"Article text: {article.get_text()[:100]}...")
                    
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    # Test MedlinePlus (Use a known disease page)
    # check_url("https://medlineplus.gov/asthma.html", "MedlinePlus Asthma")
    
    # Test the problematic page
    check_url("https://medlineplus.gov/bvitamins.html", "MedlinePlus B Vitamins")
