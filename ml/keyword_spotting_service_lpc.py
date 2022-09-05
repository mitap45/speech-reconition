import tensorflow.keras as keras
import librosa
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann
import numpy as np
from math import floor
from lpc.lpc import lpc_encode


MODEL_PATH = "model_turkish_lpc.h5"

NUM_SAMPLES_TO_CONSIDER = 22050 # 1 sec

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "ac",
        "baslat",
        "hayir",
        "evet",
        "devam",
        "ileri",
        "geri",
        "asagi",
        "dur"
    ]
    instance = None

    def predict(self, file_path):

        # extract MFCCs
        LPCs = self.preprocess(file_path) # (# segments, # coefficients)

        # convert 2d MFCCs array into 4d array ->(# samples, # segments, # coefficients, # channels)
        LPCs = LPCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(LPCs) # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract LPCs
        signal = np.array(signal)
        # normalize
        signal = 0.9 * signal / max(abs(signal))

        # resampling to 8kHz
        target_sample_rate = 8000
        target_size = int(len(signal) * target_sample_rate / sr)
        signal = resample(signal, target_size)
        sample_rate = target_sample_rate

        # 30ms Hann window
        sym = False  # periodic
        w = hann(floor(0.03 * sample_rate), sym)

        # Encode
        p = 6  # number of poles
        [A, G] = lpc_encode(signal, p, w)
        LPCs = np.concatenate((A, G))

        return LPCs.T

def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service.instance is None:
        _Keyword_Spotting_Service.instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service.instance


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/baslat.wav")

    print(f"Predicted keywords: {keyword1}")

