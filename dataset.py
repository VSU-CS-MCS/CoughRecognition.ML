#%%
import os

import scipy.io.wavfile as sp_wave

from linq_itertools import *

from domain import *
from features import *
#%%
single_cough_path = 'DataSet\\Одиночные кашли'
sex_marks = {
    'Муж': Sex.Male,
    'Жен': Sex.Female,
}
cough_type_marks = {
    'обычные': CoughType.Normal,
    'продуктивные': CoughType.Productive,
    'свистящие': CoughType.Whistling,
}
#%%
def get_dataset():
    subdirs = [subdir for subdir in os.scandir(single_cough_path) if subdir.is_dir()]
    subdir: os.DirEntry
    dataset: List[CoughData] = list()
    for subdir in subdirs:
        sex = sex_marks[
            single_or_none(
                lambda sex_mark: sex_mark in subdir.name,
                sex_marks)
            ]
        cough_type = cough_type_marks[
            single_or_none(
                lambda cough_type_mark: cough_type_mark in subdir.name,
                cough_type_marks)
        ]
        subdir_path = os.path.join(single_cough_path, subdir.name)
        audio_file: os.DirEntry
        audio_files: List[os.DirEntry] = [
            audio_file for audio_file in os.scandir(subdir_path)
            if audio_file.is_file() and audio_file.name.endswith('.wav')
        ]
        for audio_file in audio_files:
            cough_data = CoughData()
            cough_data.name = audio_file.name
            cough_data.sex = sex
            cough_data.cough_type = cough_type
            cough_data.wave_data = WaveData()
            cough_data.wave_data.framerate, cough_data.wave_data.data = sp_wave.read(audio_file.path)
            dataset.append(cough_data)
    return dataset