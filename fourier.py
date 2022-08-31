import librosa
import librosa.display
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

audio_path = "rtuk.wav"

signal, sr = librosa.load(audio_path)

plt.figure(figsize=(18,8))
plt.xlabel("Zaman (sn)")
plt.ylabel("Genlik")

librosa.display.waveshow(signal, sr, alpha=0.5, label='Zaman (sn)')
plt.show()

# ft = sp.fft.fft(signal)
# magnitude = np.absolute(ft)
# frequency = np.linspace(0, sr, len(magnitude))
#
# plt.figure(figsize=(18,8))
# plt.plot(frequency[:5000], magnitude[:5000])
# plt.xlabel("Frekans (Hz)")
# plt.ylabel("Büyüklük")
# plt.show()
#
#
