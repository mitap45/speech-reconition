import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from scipy.signal import filtfilt

def plot(signal):

    time = np.linspace(0, sr, len(signal))

    plt.plot(time, signal)
    plt.ylabel("Genlik", fontsize=15)
    plt.xlabel("Örneklem", fontsize=15)

    plt.show()


def bandPassFilter(signal):

    fs = len(signal)
    lowcut = 20.0
    highcut = 50.0

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    order = 2

    b, a = sg.butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b, a, signal, axis=0)

    return y

audio_path = "rtuk.wav"
signal, sr = librosa.load(audio_path)
signal = np.array(signal)
print(1 / sr)
print(len(signal))
print(np.amin(signal))
print(np.amax(signal))

#plot(signal)

filtered_signal = bandPassFilter(signal)

plot(filtered_signal)

exit