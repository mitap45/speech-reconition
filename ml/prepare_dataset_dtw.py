import librosa
import os
import json
import warnings
from plp import rastaplp

audio_path = "rtuk.wav"

signal, sr = librosa.load(audio_path)

PLPs = rastaplp(signal)

print(PLPs.shape)
exit