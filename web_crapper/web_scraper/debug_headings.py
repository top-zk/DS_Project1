import requests
from bs4 import BeautifulSoup

url = "https://medlineplus.gov/pneumonia.html"
res = requests.get(url)
res.raise_for_status()
soup = BeautifulSoup(res.content, "html.parser")

print("H2 headings:")
for h in soup.find_all("h2"):
    print("-", h.get_text(strip=True))

print("\nH3 headings:")
for h in soup.find_all("h3"):
    print("-", h.get_text(strip=True))

print("\nNav labels / sections present:")
for sec in soup.find_all(["section","nav","div"], attrs={"class": True}):
    classes = " ".join(sec.get("class", []))
    text = sec.get_text(strip=True)[:80]
    if "Related" in text or "Specifics" in text:
        print(classes, "->", text)