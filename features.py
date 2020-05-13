#%%
from typing import *

import numpy as np
from numpy.fft import fft
from scipy.stats import kurtosis, skew
import librosa
import librosa.feature

from domain import *
#%%
def get_time(wave_data: WaveData):
    return np.linspace(
        0, 
        len(wave_data.data) / wave_data.framerate,
        num=len(wave_data.data))

def get_spectrum(wave_data: WaveData):
    return fft(wave_data.data)

def get_mfcc(
    signal,
    framerate,
    n_mfcc=40,
    n_fft=4096):
    return librosa.feature.mfcc(
        signal,
        framerate,
        n_mfcc=n_mfcc,
        n_fft=n_fft)

def get_wave_mfcc(
    wave_data: WaveData,
    **kwargs):
    return get_mfcc(
        np.array([float(i) for i in wave_data.data]),
        wave_data.framerate,
        **kwargs)

def get_features2d(
    dataframe,
    **kwargs):
    x = list()
    for index, row in dataframe.iterrows():
        signal = np.array([float(i) for i in row.data])
        mfccs = get_mfcc(
            signal,
            row.framerate,
            **kwargs)
        features2d = [
            mfccs,
        ]
        x.append(features2d)
    return x

def get_features1d(data):
    x = list()
    for features2d in data:
        features1d = []
        for feature2d in features2d:
            features1d = np.concatenate(
                (
                    features1d,
                    np.mean(feature2d, axis=1),
                    np.min(feature2d, axis=1),
                    np.max(feature2d, axis=1),
                    np.median(feature2d, axis=1),
                    np.var(feature2d, axis=1),
                    skew(feature2d, axis=1),
                    kurtosis(feature2d, axis=1),
                ),
                axis=None)
        x.append(features1d)
    return x
