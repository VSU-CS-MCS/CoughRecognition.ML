#%%
import numpy as np
from numpy.fft import fft
from librosa.feature import mfcc

from domain import *

def get_time(wave_data: WaveData):
    return np.linspace(
        0, 
        len(wave_data.data) / wave_data.framerate,
        num=len(wave_data.data))

def get_spectrum(wave_data: WaveData):
    return fft(wave_data.data)

def get_mfcc(wave_data: WaveData, **kwargs):
    return mfcc(
        np.array([float(i) for i in wave_data.data]),
        wave_data.framerate,
        norm=kwargs.get('norm', None),
        dct_type=kwargs.get('dct_type', 2),
        n_mfcc=kwargs.get('n_mfcc', 40),
        n_fft=kwargs.get('n_fft', 2048))
