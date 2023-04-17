import requests
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm


def get_investopedia():
    with open("InvestopediaTermsList.txt", 'r') as f:
        links = f.read()
    links = links.split('\n')
    for a in tqdm(links):
        terms = BeautifulSoup(requests.get(a).content).find('main').find_all("a")
        for term in terms:
            title = term.text
            title = title.replace('/', '-').replace('%', '')
            url = BeautifulSoup(requests.get(term['href']).content)
            text = ' '.join([a.text for a in url.find_all('p')])
            with open("../data/Investopedia/" + title + ".txt", 'w') as f:
                f.write(text)


if __name__ == "__mains__":
    get_investopedia()
