#%%
from copy import *
from typing import *

import numpy as np

from dataset import *
from domain import *
#%%
def clear_noise_data():
    subdirs = [subdir for subdir in os.scandir(single_cough_path) if subdir.is_dir()]
    subdir: os.DirEntry
    for subdir in subdirs:
        subdir_path = os.path.join(single_cough_path, subdir.name)
        audio_file: os.DirEntry
        audio_files: List[os.DirEntry] = [
            audio_file for audio_file in os.scandir(subdir_path)
            if audio_file.is_file()
            and audio_file.name.endswith('.wav')
            and 'GeneratedNoise' in audio_file.name
        ]
        for audio_file in audio_files:
            os.remove(audio_file.path)
#%%
clear_noise_data()
#%%
dataset = get_dataset()
#%%
def white_gaussian_noise(std, length):
    noise_mean = 0
    return np.random.normal(
            noise_mean,
            noise_std,
            len(cough.wave_data.data))
#%%
noise_std = 100
noise_data: List[CoughData] = []
for cough in dataset:
    if 'Noise' in cough.name:
        continue

    noise_audio_amount = 3

    for i in range(noise_audio_amount):
        noise = white_gaussian_noise(
            noise_std * (i + 1),
            len(cough.wave_data.data))
        noise_cough = deepcopy(cough)
        noise_cough.name = noise_cough.name + f' GeneratedNoise {i}'
        noise_cough.wave_data.data = np.asarray(
            noise_cough.wave_data.data + noise,
            dtype=np.int16)
        noise_data.append(noise_cough)
#%%
save_dataset(noise_data)
