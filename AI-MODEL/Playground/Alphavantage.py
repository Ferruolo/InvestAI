import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm_notebook as tqdm



def get_alpha_vantage():

    soup = BeautifulSoup(requests.get("https://www.alphavantage.co/documentation/").content)
    for item in tqdm(soup.find_all('h4')):
        try:
            title = item.text
            text = item.next_sibling.next_sibling.next_sibling.next_sibling
            os.mkdir(f"../data/Alphavantage/{title}")
            with open(f"../data/Alphavantage/{title}/text.txt", 'w') as f:
                f.write(text.text)
        except FileExistsError:
            print(item.text)


if __name__ == "__main__":
    get_alpha_vantage()