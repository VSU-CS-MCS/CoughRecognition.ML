#%%
from enum import IntEnum, auto
from typing import *
#%%
class Sex(IntEnum):
    Male = 0,
    Female = 1,

class CoughType(IntEnum):
    Normal = 0,
    Productive = 1,
    Whistling = 2,

class WaveData:
    framerate: int
    data: Any
    def to_dict(self):
        return {
            'framerate': self.framerate,
            'data': self.data,
        }

class CoughData:
    name: str
    sex: Optional[Sex]
    cough_type: Optional[CoughType]
    wave_data: WaveData

    def to_dict(self):
        dictionary = {
            'name': self.name,
            'sex': self.sex,
            'cough_type': self.cough_type,
        }
        dictionary.update(self.wave_data.to_dict())
        return dictionary