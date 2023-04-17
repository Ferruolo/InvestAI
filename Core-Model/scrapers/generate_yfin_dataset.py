import json
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup



with open("./data/ticker_dataset/tickers.json") as f:
    tickers = json.load(f)

tickers = [t['TICKER'] for t in tickers]

base = "https://finance.yahoo.com/quote/"

pages = [
    "",
    "statistics",
    "profile",
    "financials",
    "analysis",
    "holders"
]
text = list()


for t in tqdm(tickers):
    for p in pages:
        page = requests.get(base + t +"/" + p)
        soup = BeautifulSoup(page.content)
        divs = soup.find_all('div')
        for div in divs:
            text.append(div.text)


labeled = list()

for t in text:
    labeled.append({
        "SENT": t,
        "CAT": "YFIN"
    })