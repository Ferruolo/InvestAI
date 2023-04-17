import json
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as ply
import string
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


num_files = 100
items_per_file = 1026
train_split = 0.80
epochs = 5
max_len = 81
w2v_size = 300
batch_split = 100
# Find Max Sentence Length


GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

train_files = [i for i in range(int(num_files * train_split))]
test_files = [i for i in range(int(num_files * train_split), num_files)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNN = nn.RNN(300, 100, 100, bidirectional=True, dtype=torch.double)
        self.flatten = nn.Flatten()
        self.lay1 = nn.Linear(16200, 1800, bias=True, dtype=torch.double)
        self.dropout = nn.Dropout()
        self.lay2 = nn.Linear(1800, 900, bias=True, dtype=torch.double)
        self.lay3 = nn.Linear(900, 503, bias=False, dtype=torch.double)

    def forward(self, x):
        x, h = self.RNN(x)
        x = self.flatten(x)
        x = F.relu(self.lay1(x))
        x = self.dropout(x)
        x = F.relu(self.lay2(x))
        x = F.relu(self.lay3(x))
        x = F.softmax(x, -2)
        return x


model = Model().to(GPU)
w2v = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)



regex_pattern = f'[^{string.ascii_lowercase}\x00-\x7F]+'


def convert_words(words):
    res = list()
    selected = list()
    for word in words:
        try:

            vectors = list()
            word = word.lower()

            word = re.sub(regex_pattern, '', word)
            for w in word.lower().split(' '):
                try:
                    vectors.append(w2v[w])
                except KeyError:
                    pass
            diff = max_len - len(vectors)
            pad = np.zeros((diff, w2v_size))
            vectors = np.stack(vectors)
            res.append(np.concatenate([vectors, pad]))
            selected.append(True)
        except ValueError:
            selected.append(False)
    res = np.stack(res)
    return res, selected


def convert_indicies(nums):
    res = list()
    for idx in nums:
        lab = np.zeros(503, )
        lab[idx] = 1
        res.append(lab)
    res = np.stack(res)
    return res




loss_list = list()
accuracy = list()
for epoch in range(epochs):
    print(f"----------------------Epoch number {epoch}----------------------")
    total_loss = 0
    epoch_accuracy = 0
    for batch_num in tqdm(train_files):

        # convert
        df = pd.read_csv(f"./data/ticker_dataset/ticker_partition_{batch_num}.csv", index_col=0)
        X, selected = convert_words(df['SENT'])
        df = df[selected]
        y = convert_indicies(df['TICK'])

        y = np.stack(y)
        X = np.stack(X)
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=445)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        for i in range(int(np.floor(X.shape[0] / batch_split)) - 1):

            torch.cuda.empty_cache()
            # print(torch.cuda.memory_usage(GPU))
            x = X[i*batch_split: (i+1)*batch_split]
            y_item = y[i*batch_split: (i+1)*batch_split]

            x = x.to(GPU)
            y_item = y_item.to(GPU)
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = criterion(y_pred, y_item)
            total_loss += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
            x = x.to(CPU)
            y_item = y_item.to(CPU)
            del x
            del y_item
            del y_pred
        num_correct = 0
        for i in range(int(np.floor(X_val.shape[0] / batch_split)) - 1):
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_usage(GPU))

            x = X_val[i*batch_split: (i+1)*batch_split]
            y_item = y_val[i*batch_split: (i+1)*batch_split]
            x = x.to(GPU)
            # y_item = y_item.to(GPU)
            y_pred = model.forward(x)
            y_pred = y_pred.to(CPU)

            for i in range(y_item.shape[0]):
                pred = y_pred[i, :].argmax()
                true = y_item[i, :].argmax()

                num_correct += pred == true
            x = x.to(CPU)
            del x
            del y_item
            del y_pred
        epoch_accuracy += (num_correct / X_val.shape[0]).cpu().detach().numpy()


    avg_loss = total_loss / (num_files * train_split)
    epoch_accuracy = epoch_accuracy / (num_files * train_split)
    loss_list.append(avg_loss)
    accuracy.append(epoch_accuracy)
    print(f"Epoch: {epoch} | Loss: {avg_loss:.02f} | Accuracy: {epoch_accuracy: .02f}")




torch.save(model.state_dict(), "./models/senToTicker.pt")
plt.figure()
plt.plot(loss_list)
plt.plot(accuracy)
plt.show()

model.to(CPU)
