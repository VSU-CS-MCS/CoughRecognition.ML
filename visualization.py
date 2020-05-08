#%%
n_dataframe = dataframe[dataframe['cough_type'] == CoughType.Normal]
p_dataframe = dataframe[dataframe['cough_type'] == CoughType.Productive]
w_dataframe = dataframe[dataframe['cough_type'] == CoughType.Whistling]
#%%
cough_data = p_dataframe.iloc[2]
wave_data = WaveData()
wave_data.data = cough_data.data
wave_data.framerate = cough_data.framerate
#%%
plot_wave(wave_data)
#%%
plot_spectrum(wave_data)
#%%
wave_mfccs = get_mfcc(wave_data, n_mfcc=40)
plot_mfccs(wave_mfccs)
