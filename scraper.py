import requests
from bs4 import BeautifulSoup

urls = ["https://credira.mintlify.app/"]
for url in urls:
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    text = soup.get_text() 
    print(text[:1000])
