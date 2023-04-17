import json
from sec_api import QueryApi, ExtractorApi
from tqdm import tqdm
import re
api_key = '9edae4de22f598b34812157615e48114785d4deb3c246b7c434bf10355794824'
extractorApi = ExtractorApi(api_key=api_key)
queryApi = QueryApi(api_key=api_key)

file_path = "./data/FinQA/train.json"


with open(file_path) as f:
    raw = json.load(f)



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
new_data = list()


for item in tqdm(raw[:10]):
    try:
        question = item['qa']['question']
        parsed = item['id'].split("/")
        ticker = parsed[0]
        year = int(parsed[1])
        page = parsed[2]
        page = page[page.index("_")+1:page.index(".")]
        page = re.sub(r'\W+', '', page)
        query = {
            "query": {
                "query_string": {
                    "query": f"ticker:\"{ticker}\" AND formType:\"10-K\" AND filedAt:[{year}-01-01 TO {year + 2}-01-01]"
                }
            }
        }
        filings = queryApi.get_filings(query)

        # Find the filing matching the given page number
        filing = filings['filings'][0]['linkToHtml']

        selected = None
        for section in sections_10k:
            text = extractorApi.get_section(filing, section, "html")
            if f"<p>{page}</p>" in text:
                print(section)
                break

        new_data.append({
            "question": question,
            "tick": ticker,
            "year": year,
            "section": selected
        })
    except IndexError:
        pass



