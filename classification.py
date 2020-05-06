#%%
from features import *
from dataset import *

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
    test_size=0.8)
X_test, X_validate, y_test, y_validate = train_test_split(
    X_test,
    y_test,
    test_size=0.5)
#%%
units = 1024
n_classes = 3
dropout = 0.1
activation = 'relu'
end_activation = 'softmax'
model = torch.nn.Sequential(
    torch.nn.Linear(feature_count, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, units),
    torch.nn.SELU(),
    torch.nn.AlphaDropout(dropout),
    torch.nn.Linear(units, n_classes),
)
#%%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
#%%
X_train_torch = torch.tensor(X_train.values).float()
X_validate_torch = torch.tensor(X_validate.values).float()
X_test_torch = torch.tensor(X_test.values).float()
y_train_torch = torch.tensor(y_train.values)
y_test_torch = torch.tensor(y_test.values)
y_validate_torch = torch.tensor(y_validate.values)
#%%
losses = []
val_losses = []
for t in range(500):
    y_train_pred = model(X_train_torch)
    loss = loss_fn(y_train_pred, y_train_torch)
    losses.append(loss)

    y_validate_pred = model(X_validate_torch)
    val_loss = loss_fn(y_validate_pred, y_validate_torch)
    val_losses.append(val_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#%%
plt.plot(losses)
#%%
plt.plot(val_losses)
#%%
y_test_pred_torch = model(X_test_torch)
test_loss = loss_fn(y_test_pred_torch, y_test_torch)
#%%
_, y_test_pred = torch.max(y_test_pred_torch, 1)
#%%
print(confusion_matrix(y_test, y_test_pred, normalize='true'))
#%%
print(classification_report(y_test, y_test_pred))
#%%
accuracy_score(y_test, y_test_pred)