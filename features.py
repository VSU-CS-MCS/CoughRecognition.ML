#%%
import numpy as np
from numpy.fft import fft
from scipy.stats import kurtosis, skew
import librosa
import librosa.feature

from domain import *

def get_time(wave_data: WaveData):
    return np.linspace(
        0, 
        len(wave_data.data) / wave_data.framerate,
        num=len(wave_data.data))

def get_spectrum(wave_data: WaveData):
    return fft(wave_data.data)

def get_mfcc(wave_data: WaveData, **kwargs):
    return 

def get_features2d(dataframe):
    x = list()
    for index, row in dataframe.iterrows():
        signal = np.array([float(i) for i in row.data])
        mfccs = librosa.feature.mfcc(
            signal,
            row.framerate,
            n_mfcc=40)
        chromagram = librosa.feature.chroma_stft(
            signal,
            row.framerate)
        features2d = [mfccs, chromagram]
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
