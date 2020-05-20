#%%
import matplotlib.pyplot as plt
import seaborn as sns

import librosa.display

from features import *
#%%
def plot_wave(wave_data: WaveData):
    time = get_time(wave_data)
    plt.plot(time, wave_data.data)

def plot_spectrum(wave_data: WaveData):
    spectrum = get_spectrum(wave_data)
    plt.plot(spectrum)

def plot_mfccs(wave_mfccs, framerate):
    librosa.display.specshow(wave_mfccs, sr=framerate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_confusion(confusion):
    labels = ['Dry', 'Wet', 'Whistling']
    sns.heatmap(confusion, annot=True,
        xticklabels=labels, yticklabels=labels)
    plt.show()
