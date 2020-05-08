#%%
from typing import *

from datetime import datetime
import os

import pandas as pd

import numpy as np
from numpy.fft import fft
import scipy.signal as sp_signal

from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import *

import torch

import matplotlib.pyplot as plt

from charts import *
from dataset import *
from features import *
#%%
seed = 666
#%%
dataset = get_dataset()
#%%
dataframe = pd.DataFrame.from_records([w.to_dict() for w in dataset])
#%%
x2d = get_features2d(dataframe)
#%%
x1d = get_features1d(x2d)
feature_count = len(x1d[0])
#%%
X = pd.DataFrame(x1d)
y = dataframe['cough_type']
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=seed)
X_test, X_validate, y_test, y_validate = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=seed)
#%%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#%%
X_train_torch = torch.tensor(X_train.values).float().to(device)
X_validate_torch = torch.tensor(X_validate.values).float().to(device)
X_test_torch = torch.tensor(X_test.values).float().to(device)
y_train_torch = torch.tensor(y_train.values).to(device)
y_test_torch = torch.tensor(y_test.values).to(device)
y_validate_torch = torch.tensor(y_validate.values).to(device)
#%%
units = 64
n_classes = 3
dropout = 0.1
hidden_layers_amount = 8

model_args = [
    torch.nn.Linear(feature_count, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
]

for layer_index in range(hidden_layers_amount - 1):
    model_args.extend([
        torch.nn.Linear(units, units),
        torch.nn.SELU(),
        torch.nn.AlphaDropout(dropout)
    ])

model_args.append(torch.nn.Linear(units, n_classes))
model = torch.nn.Sequential(*model_args).to(device)
#%%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
#%%
train_losses = []
train_accs = []
val_losses = []
val_accs = []
model_dir = 'output'
model_path = os.path.join(model_dir, f'model.pt')
for epoch in range(3000):
    y_train_pred_torch = model(X_train_torch)
    train_loss = loss_fn(y_train_pred_torch, y_train_torch)
    train_losses.append(train_loss.item())

    y_validate_pred_torch = model(X_validate_torch)
    val_loss = loss_fn(y_validate_pred_torch, y_validate_torch)
    val_losses.append(val_loss.item())

    _, y_train_pred = torch.max(y_train_pred_torch.cpu(), 1)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_accs.append(train_acc)

    _, y_validate_pred = torch.max(y_validate_pred_torch.cpu(), 1)
    val_acc = accuracy_score(y_validate, y_validate_pred)
    val_accs.append(val_acc)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if (val_acc >= np.max(val_accs)):
        torch.save(model.state_dict(), model_path)
        print(f'{epoch} saved')

    if (epoch % 100 == 99):
        print(f'{epoch} {train_loss} {train_acc} {val_loss} {val_acc}')
#%%
model.load_state_dict(torch.load(model_path))
model.eval()
#%%
plt.plot(train_losses)
#%%
plt.plot(val_losses)
#%%
y_test_pred_torch = model(X_test_torch)
test_loss = loss_fn(y_test_pred_torch, y_test_torch)
_, y_test_pred = torch.max(y_test_pred_torch, 1)
y_test_pred_cpu = y_test_pred.cpu()
#%%
print(confusion_matrix(y_test, y_test_pred_cpu, normalize='true'))
#%%
print(classification_report(y_test, y_test_pred_cpu))
#%%
accuracy_score(y_test, y_test_pred_cpu)