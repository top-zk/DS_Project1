import requests
from bs4 import BeautifulSoup

URL = "https://medlineplus.gov/ency/encyclopedia_A.htm"

def get_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

soup = get_soup(URL)
if soup:
    with open("debug_output.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print("HTML saved to debug_output.html")