import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "rtuk.wav"

signal, sr = librosa.load(audio_path)

plt.figure(figsize=(18,8))
plt.ylabel("Genlik", fontsize=15)
librosa.display.waveshow(signal, sr, alpha=0.5)
plt.title("Dijital Dalga Formu")
plt.xlabel("Zaman", fontsize=15)

plt.show()
