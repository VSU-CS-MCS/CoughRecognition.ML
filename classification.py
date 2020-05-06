#%%
from features import *
from dataset import *

from datetime import datetime
import os

import pandas as pd

import numpy as np
from numpy.fft import fft
import scipy.signal as sp_signal
from scipy.stats import kurtosis, skew

import librosa
from librosa.feature import mfcc
import librosa.display

from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import *

import tensorflow as tf
%load_ext tensorboard
from tensorflow import keras

import matplotlib.pyplot as plt

from charts import *
#%%
dataset = get_dataset()
#%%
dataframe = pd.DataFrame.from_records([w.to_dict() for w in dataset])
#%%
mfccs = 'mfccs'
dataframe[mfccs] = dataframe.apply(lambda it: get_mfcc(it), axis=1)
#%%
x = list()
for index, row in dataframe.iterrows():
    mfccs_features = dict()
    mfccs_min = np.min(row[mfccs], axis=1)
    mfccs_max = np.max(row[mfccs], axis=1)
    mfccs_median = np.median(row[mfccs], axis=1)
    mfccs_mean = np.mean(row[mfccs], axis=1)
    mfccs_variance = np.var(row[mfccs], axis=1)
    mfccs_skeweness = skew(row[mfccs], axis=1)
    mfccs_kurtosis = kurtosis(row[mfccs], axis=1)
    mfccs_features = np.concatenate([
        mfccs_min,
        mfccs_max,
        mfccs_median,
        mfccs_mean,
        mfccs_variance,
        mfccs_skeweness,
        mfccs_kurtosis])
    x.append(mfccs_features)
#%%
X = pd.DataFrame(x)
y = dataframe['cough_type']
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3)
X_validate, X_test, y_validate, y_test = train_test_split(
    X_test,
    y_test,
    test_size=0.5)
#%%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(3)
])
#%%
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
#%%
history: keras.callbacks.History = model.fit(
    x=X_train.values,
    y=y_train.values,
    epochs=100,
    validation_data=(X_validate.values, y_validate.values))
#%%
Y_pred = model.predict(X)
y_pred = np.argmax(Y_pred, axis=1)
#%%
print(confusion_matrix(y, y_pred, normalize='true'))
#%%
print(classification_report(y, y_pred))
#%%
accuracy_score(y, y_pred)
#%%
Y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(Y_test_pred, axis=1)
#%%
print(confusion_matrix(y_test, y_test_pred, normalize='true'))
#%%
print(history)