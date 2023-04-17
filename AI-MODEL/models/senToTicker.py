import json
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as ply


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
with open('./data/GPT-ticker-dataset/sentence-to_ticker.json', 'r') as f:
    data = json.load(f)

with open('./data/tickers.json', 'r') as f:
    tickers = json.load(f)

tickers = [t['TICKER'] for t in tickers]
tick_to_idx = dict()
for i in range(len(tickers)):
    t = tickers[i]
    tick_to_idx[t] = i

w2v = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)

missed_words = list()

X = list()
y = list()
for item in tqdm(data):
    sent = item['SENT']
    vectors = list()
    for w in sent.lower().split(' '):
        try:
            vectors.append(w2v[w])
        except KeyError:
            missed_words.append(w)
    target = np.zeros(len(tickers))
    for t in item['ENT']:
        try:
            target[tick_to_idx[t]] = 1
        except KeyError:
            pass
    X.append(np.stack(vectors))
    y.append(target)

del w2v

y = np.stack(y)

max_len = 0
for i in X:
    if i.shape[0] > max_len:
        max_len = i.shape[0]

for x in range(len(X)):
    diff = max_len - X[x].shape[0]
    pad = np.zeros((diff, 300))
    X[x] = np.concatenate([X[x], pad])

X = np.stack(X)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNN = nn.RNN(300, 100, 100, bidirectional=True, dtype=torch.double)
        self.flatten = nn.Flatten()
        self.lay1 = nn.Linear(3600, 1800, bias=True, dtype=torch.double)
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


model = Model()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=445)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

epochs = 5

model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_train[:100]
y_val = y_train[:100]
X_train = X_train[100:]
y_train = y_train[100:]


batch_size = 100

loss_list = list()
accuracy = list()

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        x = X_train[i:i+batch_size, :]
        y_iter = y_train[i:i+batch_size, :]
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = criterion(y_pred, y_iter)
        total_loss += loss
        loss.backward()
        optimizer.step()
        x.to("cpu")
        y_iter.to("cpu")

    y_pred = model.forward(X_val)

    num_correct = 0
    for i in range(y_val.shape[0]):
        pred = y_pred[i, :].argmax()
        true = y_val[i, :].argmax()

        num_correct += pred == true
    avg_loss = (total_loss / X_train.shape[0]).cpu().detach().numpy()
    epoch_accuracy = (num_correct / y_val.shape[0]).cpu().detach().numpy()
    loss_list.append(avg_loss)
    accuracy.append(epoch_accuracy)

    print(f"Epoch: {epoch} | Loss: {avg_loss} | Accuracy: {epoch_accuracy: .02f}")

plt.figure()
plt.plot(loss_list)
plt.plot(accuracy)
plt.show()


torch.save(model.state_dict(), "./models/senToTicker.pt")

X_train.to('cpu')
y_train.to('cpu')

X_test = X_test.to(device)
y_test = y_test.to(device)
y_pred = model.forward(X_test)
test_loss = criterion(y_pred, y_test)
print(f"Loss: {test_loss}")

num_correct = 0
for i in range(y_test.shape[0]):
    pred = y_pred[i, :].argmax()
    true = y_test[i, :].argmax()

    num_correct += pred == true

print(f"Accuracy {(num_correct/ y_test.shape[0])*100: .02f}%")