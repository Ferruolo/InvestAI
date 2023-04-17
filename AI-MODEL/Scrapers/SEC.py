from sec_api import QueryApi, ExtractorApi
import pandas as pd
import json
from tqdm import tqdm_notebook as tqdm

api_key = '9edae4de22f598b34812157615e48114785d4deb3c246b7c434bf10355794824'
extractorApi = ExtractorApi(api_key=api_key)

sections_10k = [
    '1',
    '1A',
    '1B',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '7A',
    '8',
    '9',
    '9A',
    '9B',
    '10',
    '11',
    '12',
    '13',
    '14'
]

sections_10q = [
    'part1item1',
    'part1item2',
    'part1item3',
    'part1item4',
    'part2item1',
    'part2item1a',
    'part2item2',
    'part2item3',
    'part2item4',
    'part2item5',
    'part2item6',
]

sections_8k = [
    "1-1",
    "1-2",
    "1-3",
    "1-4",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "2-5",
    "2-6",
    "3-1",
    "3-2",
    "3-3",
    "4-1",
    "4-2",
    "5-1",
    "5-2",
    "5-3",
    "5-4",
    "5-5",
    "5-6",
    "5-7",
    "5-8",
    "6-1",
    "6-2",
    "6-3",
    "6-4",
    "6-5",
    "6-6",
    "6-10",
    "7-1",
    "8-1",
    "9-1"
]


def get_ticker(ticker):
    queryApi = QueryApi(api_key=api_key)
    query = {
        "query": {"query_string": {
            "query": f"ticker:{ticker}",
        }},
        "from": "0",
        "size": "1000",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = queryApi.get_filings(query)
    metadata = pd.DataFrame.from_records(response['filings'])
    try:
        url = metadata[metadata['formType'] == '10-K']['linkToHtml'].reset_index()['linkToHtml'][0]
        for section in sections_10k:
            text = extractorApi.get_section(url, section, "text")
            with open(f"../data/SEC-EDGAR/10-K/{section}/{ticker}.txt", 'w') as f:
                f.write(text)
    except ValueError:
        print(ticker)

    try:
        url = metadata[metadata['formType'] == '10-Q']['linkToHtml'].reset_index()['linkToHtml'][0]
        for section in sections_10q:
            text = extractorApi.get_section(url, section, "text")
            with open(f"../data/SEC-EDGAR/10-Q/{section}/{ticker}.txt", 'w') as f:
                f.write(text)
    except ValueError:
        print(ticker)

    try:
        url = metadata[metadata['formType'] == '8-K']['linkToHtml'].reset_index()['linkToHtml'][0]
        for section in sections_8k:
            text = extractorApi.get_section(url, section, "text")
            with open(f"../data/SEC-EDGAR/8-K/{section}/{ticker}.txt", 'w') as f:
                f.write(text)
    except ValueError:
        print(ticker)


def get_all_tickers():
    with open("../data/tickers.json", 'r') as f:
        tickers = json.load(f)

    tickers = [a['TICKER'] for a in tickers]
    for ticker in tickers:
        get_ticker(tickers)




if __name__ == "__main__":
    get_all_tickers()