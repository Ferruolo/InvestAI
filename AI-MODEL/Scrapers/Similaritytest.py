import gensim
import re
import os

import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.optim.lr_scheduler as lr_scheduler
documents = []  # Format {"Name", "Text"}

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


base_dir = "./data"
print("Loading AlphaVantage")
for d in tqdm(os.listdir(base_dir + f"/Alphavantage")):
    if re.match("^[a-zA-Z0-9]", d):
        with open(base_dir + f"/Alphavantage/{d}/text.txt", 'r') as f:
            file = f.read()
            clean = re.sub(r'[^\w\s]', '', file)
            clean = re.sub(r'[^\x00-\x7F]', '', clean)
            clean = clean.replace("\n", "")
            clean = clean.lower()
            documents.append({"NAME": "Alphavantage", "DATA": clean})

print("Loading Investopedia")
for d in tqdm(os.listdir(base_dir + f"/Investopedia")):
    if re.match("^[a-zA-Z0-9]", d):
        with open(base_dir + f"/Investopedia/{d}", 'r') as f:
            file = f.read()
            clean = re.sub(r'[^\w\s]', '', file)
            clean = re.sub(r'[^\x00-\x7F]', '', clean)
            clean = clean.replace("\n", "")
            clean = clean.lower()
            documents.append({"NAME": "Investopedia", "DATA": clean})

print("Loading SEC-EDGAR")
base_dir += '/SEC-EDGAR'
for ft in tqdm(os.listdir(base_dir)):
    for section in (os.listdir(base_dir + f"/{ft}")):
        if re.match("^[a-zA-Z0-9]", section):
            for file in (os.listdir(base_dir + f"/{ft}/{section}")):
                if re.match("^[a-zA-Z0-9]", file):
                    with open(base_dir + f"/{ft}/{section}/{file}", 'r') as f:
                        text = f.read()
                        clean = re.sub(r'[^\w\s]', '', file)
                        clean = re.sub(r'[^\x00-\x7F]', '', clean)
                        clean = clean.replace("\n", "")
                        clean = clean.lower()
                        documents.append({"NAME": f"{ft}", "DATA": clean})


print("Vectorizing Items")
TFIDF = TfidfVectorizer()

labels = pd.Series([x['NAME'] for x in documents])

labels = labels.drop_duplicates().reset_index()[0]

lookup = dict()
for i in range(labels.size):
    lookup[labels[i]] = i

TFIDF = TFIDF.fit([x['DATA'] for x in documents])
X = TFIDF.transform([x['DATA'] for x in documents])
X = X.toarray()
initial_size = X.shape[1]
y = np.zeros((len(documents), labels.size))
for i in range(len(documents)):
    y[i, lookup[documents[i]['NAME']]] = 1


over_min = np.average(X, axis=0) >= 0.0001
X = X[:, over_min]
print(f"Percentage_Left {X.shape[1] / initial_size:.02f}")
print(X.shape)
print(y.shape)
print("Setting up training")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=445, shuffle=True)



class Model(nn.Module):
    def __init__(self):#X_SHAPE:(7807, 31469) Y_SHAPE: 7807, 6484
        super().__init__()
        self.lin1 = nn.Linear(9106, 4500, dtype=torch.float64)
        self.lin2 = nn.Linear(4500, 2250, dtype=torch.float64)
        # self.lin3 = nn.Linear(2250, 1125, dtype=torch.float64)
        self.lin4 = nn.Linear(2250, 552, dtype=torch.float64)
        self.lin5 = nn.Linear(552, 100, dtype=torch.float64)
        self.lin6 = nn.Linear(100, 20, dtype=torch.float64)
        self.lin7 = nn.Linear(20, 5, dtype=torch.float64)

        self.layers1 = [
            self.lin1,
            self.lin2,
            # self.lin3,
            self.lin4,

        ]

        self.layers2 = [

            self.lin5,
            self.lin6,
            self.lin7
        ]

        self.dropout = nn.Dropout(0.25)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers1 + self.layers2:
            c_in = layer.weight.size(1)
            nn.init.normal_(layer.weight, 0.0, 1/np.sqrt(c_in))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for layer in self.layers1:
            x = F.relu(layer(x))
        x = self.dropout(x)
        for layer in self.layers2[:-1]:
            x = F.relu(layer(x))
        return self.layers2[-1](x)


model = Model().to(device)


print("Training Model")
# Train:

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = X_train[:100]
y_val = y_train[:100]
X_train = X_train[100:]
y_train = y_train[100:]

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
epochs = 5


batch_size = 250

X_val = X_val.to(device)
y_val = y_val.to(device)

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        x = X_train[i:i+batch_size, :].to(device)
        y_iter = y_train[i:i+batch_size, :].to(device)
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = criterion(y_pred, y_iter)
        total_loss += loss
        loss.backward()
        optimizer.step()
        x.to("cpu")
        y_iter.to("cpu")
        scheduler.step()
    y_pred = model.forward(X_val)

    num_correct = 0
    for i in range(y_val.shape[0]):
        pred = y_pred[i, :].argmax()
        true = y_val[i, :].argmax()

        num_correct += pred == true
    print(f"Epoch: {epoch} | Loss: {total_loss / X_train.shape[0]} | Accuracy: {num_correct/y_val.shape[0]: .02f}")

torch.save(model.state_dict(), "./models/Similarity.pt")

X_train.to('cpu')
y_train.to('cpu')

X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
y_pred = model.forward(X_test)
test_loss = criterion(y_pred, y_test)
print(f"Loss: {test_loss}")

num_correct = 0
for i in range(y_test.shape[0]):
    pred = y_pred[i, :].argmax()
    true = y_test[i, :].argmax()

    num_correct += pred == true

print(f"Accuracy {num_correct/ y_test.shape[0]: .04f}")
