import requests
from bs4 import BeautifulSoup
import json


def load_ticker_file():
    soup = BeautifulSoup(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").content, parser="lxml")

    table = soup.find('table')

    rows = table.find_all('tr')

    data = list()

    for row in rows[1:]:
        info = row.find_all('td')
        data.append({
            "TICKER": info[0].text.replace("\n", ""),
            "NAME": info[0].text.replace("\n", "")
        })


    with open("../data/tickers.json", 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    load_ticker_file()