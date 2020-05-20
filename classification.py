#%%
from typing import *
from functools import reduce
from itertools import product

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set_style('darkgrid')

from charts import *
from dataset import *
from features import *
from model import *
#%%
dataset = get_dataset()
#%%
dataframe = pd.DataFrame.from_records([w.to_dict() for w in dataset])
#%%
noise_index = dataframe['name'].str.contains('GeneratedNoise')
df_noise = dataframe[noise_index]
df_raw = dataframe[~noise_index]
raw_to_noise = {}
for index, row in df_raw.iterrows():
    raw_to_noise[index] = df_noise['name'].str.contains(f"{row['name']} GeneratedNoise")
#%%
y = dataframe['cough_type']
#%%
def get_features(df, **kwargs):
    x2d = get_features2d(df, **kwargs)
    x1d = get_features1d(x2d)
    feature_count = len(x1d[0])
    X = pd.DataFrame(x1d)
    return X, feature_count
#%%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#%%
test_size = 0.1
validate_size = 0.1
def dataframe_split(df, seed = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size+validate_size,
        random_state=seed)
    df_test, df_validate = train_test_split(
        df_test,
        test_size=validate_size,
        random_state=seed)
    return df_train, df_validate, df_test

def X_y_split(X, y, seed = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size+validate_size,
        random_state=seed)
    X_test, X_validate, y_test, y_validate = train_test_split(
        X_test,
        y_test,
        test_size=validate_size,
        random_state=seed)
    return X_train, X_validate, X_test, y_train, y_validate, y_test
#%%
def train_test(
    checkpoint_path,
    X_train, X_validate, X_test,
    y_train, y_validate, y_test,
    feature_count,
    seed = None,
    silent = True,
    **kwargs):
    checkpoint_path = f'{checkpoint_path}.pt'
    np.random.seed(seed)
    manual_seed = seed != None
    torch.backends.cudnn.deterministic = manual_seed
    torch.backends.cudnn.benchmark = not manual_seed
    if (manual_seed):
        torch.manual_seed(seed)
    else:
        torch.seed()

    X_train_torch = torch.tensor(X_train.values).float().to(device)
    X_validate_torch = torch.tensor(X_validate.values).float().to(device)
    X_test_torch = torch.tensor(X_test.values).float().to(device)
    y_train_torch = torch.tensor(y_train.values).to(device)
    y_test_torch = torch.tensor(y_test.values).to(device)
    y_validate_torch = torch.tensor(y_validate.values).to(device)

    model = get_ffn_net(feature_count, **kwargs).to(device)

    weights = torch.FloatTensor([1.0, 3.0, 3.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weights)

    optimizer = torch.optim.AdamW(model.parameters())

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(1000):
        y_train_pred_torch = model(X_train_torch)
        train_loss = loss_fn(y_train_pred_torch, y_train_torch)
        train_losses.append(train_loss.item())

        _, y_train_pred = torch.max(y_train_pred_torch.cpu(), 1)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_accs.append(train_acc)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_validate_pred_torch = model(X_validate_torch)
            val_loss = loss_fn(y_validate_pred_torch, y_validate_torch)
            val_losses.append(val_loss.item())

            _, y_validate_pred = torch.max(y_validate_pred_torch.cpu(), 1)
            val_acc = accuracy_score(y_validate, y_validate_pred)
            val_accs.append(val_acc)

        if (val_loss.item() <= np.min(val_losses)):
            torch.save(model.state_dict(), checkpoint_path)
            if (silent):
                continue
            print(f'{epoch} saved')
            print(f'{epoch} Loss {train_loss} {val_loss}')
            print(f'{epoch} Accuracy {train_acc} {val_acc}')
        elif (epoch % 100 == 99):
            if (silent):
                continue
            print(f'{epoch} Loss {train_loss} {val_loss}')
            print(f'{epoch} Accuracy {train_acc} {val_acc}')

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    y_test_pred_torch = model(X_test_torch)
    test_loss = loss_fn(y_test_pred_torch, y_test_torch)
    _, y_test_pred = torch.max(y_test_pred_torch, 1)
    y_test_pred_cpu = y_test_pred.cpu()

    confusion = confusion_matrix(y_test, y_test_pred_cpu, normalize='true')
    accuracy = accuracy_score(y_test, y_test_pred_cpu)

    if not silent:
        plt.plot(train_losses, 'r', val_losses, 'b')
        plt.legend(['Train', 'Validate', 'Test'])
        plt.show()

        plt.plot(train_accs, 'r', val_accs, 'b')
        plt.legend(['Train', 'Validate', 'Test'])
        plt.show()

        plot_confusion(confusion)

        print(classification_report(y_test, y_test_pred_cpu))
    return confusion, accuracy, test_loss.item()
#%%
def train_test_multiple(
    train_amount,
    checkpoint_path,
    X_train, X_validate, X_test,
    y_train, y_validate, y_test,
    feature_count,
    **kwargs):
    results = [train_test(
        f'{checkpoint_path}_{i}',
        X_train, X_validate, X_test,
        y_train, y_validate, y_test,
        feature_count,
        **kwargs) for i in range(train_amount)]
    confusions = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    losses = [result[2] for result in results]
    return confusions, accuracies, losses
#%%
def print_results(confusions, accuracies, losses):
    print(f'{np.mean(accuracies)} +-{np.std(accuracies)}')
    print(f'{np.mean(losses)} +-{np.std(losses)}')
    loss_min_index = np.argmin(losses)
    print(f'{accuracies[loss_min_index]} {losses[loss_min_index]}')
    sns.heatmap(confusions[loss_min_index], annot=True)
#%%
def get_param_combinations(param_groups):
    args_set = list()
    for param_group in param_groups:
        param_keys = [key for key in param_group]
        params_sets = product(*[value for key, value in param_group.items()])
        args_set.extend([
            {
                param_keys[index]: param
                for index, param in enumerate(param_set)
            }
            for param_set in params_sets
        ])
    return args_set
#%%
feature_param_groups = [
    {
        'n_mfcc': [40],
    }
]
model_param_groups = [
    {
        'units': [64, 128],
        'dropout': [0, 0.1, 0.2],
        'layers': [4, 8, 16],
    }
]
model_param_combinations = get_param_combinations(model_param_groups)
feature_param_combinations = get_param_combinations(feature_param_groups)
split_amount = 1
train_amount = 1
model_dir = 'output'
#%%
feature_params_cache = {}
for feature_param_combination in feature_param_combinations:
    feature_index = tuple(feature_param_combination)
    feature_params_cache[feature_index] = get_features(dataframe, **feature_param_combination)
#%%
split_cache = {}
for split_i in range(split_amount):
    split_cache[split_i] = dataframe_split(df_raw)
    df_train, df_validate, df_test = split_cache[split_i]
    for index, row in df_train.iterrows():
        df_train = df_train.append(df_noise[raw_to_noise[index]])
    split_cache[split_i] = (df_train, df_validate, df_test)
#%%
results_df = pd.DataFrame()
#%%
silent = True
for split_i in range(split_amount):
    df_train, df_validate, df_test = split_cache[split_i]
    for feature_param_combination in feature_param_combinations:
        feature_index = tuple(feature_param_combination)
        for model_param_combination in model_param_combinations:
            model_param_path = f'{model_param_combination}' \
                .replace('{', '') \
                .replace('}', '') \
                .replace("'", '') \
                .replace(":", '') \
                .replace(' ', '')
            feature_param_path = f'{feature_param_combination}' \
                .replace('{', '') \
                .replace('}', '') \
                .replace("'", '') \
                .replace(":", '') \
                .replace(' ', '')
            checkpoint_path = os.path.join(model_dir, f'model_{split_i}_{feature_param_path}_{model_param_path}')
            X, feature_count = feature_params_cache[feature_index]

            def get_split(df):
                indexes = list(df.index.values)
                return X.loc[indexes, :], y.loc[indexes]

            X_train, y_train = get_split(df_train)
            X_validate, y_validate = get_split(df_validate)
            X_test, y_test = get_split(df_test)

            confusions, accuracies, losses = train_test_multiple(
                train_amount,
                checkpoint_path,
                X_train, X_validate, X_test,
                y_train, y_validate, y_test,
                feature_count,
                silent=silent,
                **model_param_combination)
            for i in range(len(losses)):
                result = {
                    'confusion': confusions[i],
                    'accuracy': accuracies[i],
                    'loss': losses[i],
                    'split': split_i,
                }
                result.update(**feature_param_combination)
                result.update(**model_param_combination)
                results_df = results_df.append(
                    [result],
                    ignore_index=True)
#%%
results_df['split'] = results_df['split'].astype(str)
#%%
min_indexes = results_df \
    .groupby(['split', 'layers'])['loss'] \
    .idxmin()
results_df_min = results_df.loc[min_indexes,]
px.scatter_3d(
    results_df_min,
    x='layers',
    y='units',
    z='loss',
    color='split',
    symbol='dropout',
    color_discrete_sequence=px.colors.sequential.Rainbow
)
#%%
min_indexes = results_df \
    .groupby(['split'])['loss'] \
    .idxmin()
results_df_min = results_df.loc[min_indexes,]
px.scatter_3d(
    results_df_min,
    x='layers',
    y='units',
    z='loss',
    color='split',
    symbol='dropout',
    color_discrete_sequence=px.colors.sequential.Rainbow
)
#%%
min_index = results_df['loss'].idxmin()
min_item = results_df.iloc[min_index]
plot_confusion(min_item['confusion'])