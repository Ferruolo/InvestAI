import pandas as pd
import pickle
from tqdm import tqdm
import json
from gensim.models import KeyedVectors
import numpy as np

num_dataframes = 1000


def get_dataframe(file_num: int):
    df = pd.read_csv(f"./data/ticker_dataset/ticker_partition_{file_num}.csv")
    return df


def apply_all(column_key, func):
    for i in range(num_dataframes):
        df = get_dataframe(i)
        df[column_key].apply(func)


class find_max_len:
    def __init__(self):
        self.max_len = 0

    def forward(self, item):
        if len(item) > self.max_len:
            self.max_len = len(item)



max_len = find_max_len()
apply_all('SENT', max_len.forward)

max_len = max_len.max_len


def write_pickles(X, y, i):
    with open(f"./data/ticker_dataset/X_partition_{i}.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open(f"./data/ticker_dataset/y_partition_{i}.pkl", 'wb') as f:
        pickle.dump(y, f)


w2v = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)


for i in tqdm(range(num_dataframes)):
    df = get_dataframe(i)
    X = list()
    y = list()

    for row in df.itertuples():
        try:
            word_vecs = list()

            for w in row[2].split(' '):
                try:
                    word_vecs.append(w2v[w])
                except KeyError:
                    pass
            pad = np.zeros(300)
            diff = max_len - len(word_vecs)
            for j in range(diff):
                word_vecs.append(pad)
            word_vecs = np.stack(word_vecs)
            label = np.zeros(503, )
            label[row[3]] = 1
            X.append(word_vecs)
            y.append(label)


        except KeyError:
            pass
    write_pickles(X, y, i)

del w2v

tickers = list()

for i in tqdm(range(50)):
    df = get_dataframe(i)
    new_tickers = df['stock'].drop_duplicates()
    for t in new_tickers:
        tickers.append(t)

