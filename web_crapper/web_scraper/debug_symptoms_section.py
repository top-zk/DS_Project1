import requests
from bs4 import BeautifulSoup

url = "https://medlineplus.gov/pneumonia.html"
res = requests.get(url)
res.raise_for_status()
soup = BeautifulSoup(res.content, "html.parser")

summary = soup.find('div', id='topic-summary')
h3 = summary.find('h3', string='What are the symptoms of pneumonia?') if summary else None
print('Found summary:', bool(summary), 'Found h3:', bool(h3))
if h3:
    tag = h3.find_next_sibling()
    count = 0
    while tag and count < 20 and tag.name != 'h3':
        print('TAG:', tag.name)
        print((tag.get_text(strip=True)[:500]).replace('\n',' '))
        tag = tag.find_next_sibling()
        count += 1