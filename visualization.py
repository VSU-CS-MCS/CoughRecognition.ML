#%%
from typing import *

import sklearn.preprocessing

from charts import *
#%%
n_dataframe = dataframe[dataframe['cough_type'] == CoughType.Normal]
p_dataframe = dataframe[dataframe['cough_type'] == CoughType.Productive]
w_dataframe = dataframe[dataframe['cough_type'] == CoughType.Whistling]
#%%
cough_data = p_dataframe.iloc[0]
wave_data = WaveData()
wave_data.data = cough_data.data
wave_data.framerate = cough_data.framerate
#%%
plot_wave(wave_data)
#%%
plot_spectrum(wave_data)
#%%
wave_mfccs = get_mfcc(wave_data)
plot_mfccs(wave_mfccs)
#%%
wave_mfccs = sklearn.preprocessing.scale(wave_mfccs, axis=1)
plot_mfccs(wave_mfccs)
