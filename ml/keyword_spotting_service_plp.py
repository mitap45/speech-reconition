import tensorflow.keras as keras
import librosa
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann
import numpy as np
from math import floor
from plp import rastaplp


MODEL_PATH = "model_turkish_plp.h5"

NUM_SAMPLES_TO_CONSIDER = 22050 # 1 sec

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "ac",
        "asagi",
        "baslat",
        "devam",
        "dur",
        "evet",
        "geri",
        "hayir",
        "ileri",
        "iptal",
        "kapa",
        "sag",
        "sol",
        "yukari"
    ]
    instance = None

    def predict(self, file_path):

        # extract MFCCs
        PLPs = self.preprocess(file_path) # (# segments, # coefficients)

        # convert 2d MFCCs array into 4d array ->(# samples, # segments, # coefficients, # channels)
        PLPs = PLPs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(PLPs) # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract LPCs
        signal = np.array(signal)
        PLPs = rastaplp(signal)

        return PLPs.T

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

