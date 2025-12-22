import requests
from bs4 import BeautifulSoup

url = "https://medlineplus.gov/pneumonia.html"
res = requests.get(url)
res.raise_for_status()
soup = BeautifulSoup(res.content, "html.parser")

def show_section(title):
    h2 = soup.find(lambda t: t.name == 'h2' and t.get_text(strip=True) == title)
    print("Found:", bool(h2), title)
    if not h2:
        return
    sib = h2.find_next_sibling()
    i = 0
    while sib and not (sib.name == 'h2') and i < 20:
        print("- sib:", sib.name, sib.get("class"))
        links = sib.find_all('a')
        if links:
            print("  links:",[a.get_text(strip=True) for a in links][:5])
        i += 1
        sib = sib.find_next_sibling()

show_section("Related Issues")
show_section("Related Health Topics")