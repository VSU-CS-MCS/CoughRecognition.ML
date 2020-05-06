#%%
from copy import *
from typing import *

import numpy as np

from dataset import *
from domain import *
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

    noise_audio_amount = 1 if cough.cough_type == CoughType.Normal else 10

    for i in range(noise_audio_amount):
        noise = white_gaussian_noise(
            noise_std,
            len(cough.wave_data.data))
        noise_cough = deepcopy(cough)
        noise_cough.name = noise_cough.name + f' Noise {i}'
        noise_cough.wave_data.data = np.asarray(
            noise_cough.wave_data.data + noise,
            dtype=np.int16)
        noise_data.append(noise_cough)
#%%
save_dataset(noise_data)
