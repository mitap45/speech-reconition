import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

scale_file = "scale.wav"
rtuk_file = "rtuk.wav"

scale, _ = librosa.load(scale_file)
rtuk, sr = librosa.load(rtuk_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
S_rtuk = librosa.stft(rtuk, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

Y_scale = np.abs(S_scale) ** 2
Y_rtuk = np.abs(S_rtuk) ** 2

def plot_spectogram(Y, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length = hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

Y_log_scale = librosa.power_to_db(Y_scale)
Y_log_rtuk = librosa.power_to_db(Y_rtuk)
plot_spectogram(Y_log_rtuk, sr, HOP_SIZE)
plt.show()