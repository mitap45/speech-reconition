import scipy.io.wavfile
import numpy as np
from math import floor
import scipy.signal as signal
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann
from numpy.random import randn
import lpc
import librosa

[sample_rate, amplitudes] = scipy.io.wavfile.read('speech.wav')

amplitudes = np.array(amplitudes)
# normalize
amplitudes = 0.9*amplitudes/max(abs(amplitudes))

# resampling to 8kHz
target_sample_rate = 8000
target_size = int(len(amplitudes)*target_sample_rate/sample_rate)
amplitudes = resample(amplitudes, target_size)
sample_rate = target_sample_rate

# 30ms Hann window
sym = False # periodic
w = hann(floor(0.03*sample_rate), sym)

# Encode
p = 6 # number of poles
[A, G] = lpc.lpc_encode(amplitudes, p, w)

# Print stats
original_size = len(amplitudes)
model_size = A.size + G.size
print('Original signal size:', original_size)
print('Encoded signal size:', model_size)
print('Data reduction:', original_size/model_size)

xhat = lpc.lpc_decode(A, G, w)

scipy.io.wavfile.write("example.wav", sample_rate, xhat)
print('done')
