import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"

NUM_SAMPLES_TO_CONSIDER = 22050 # 1 sec

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "dataset\\bed",
        "dataset\\bird",
        "dataset\\cat",
        "dataset\\dog",
        "dataset\\down",
        "dataset\\eight",
        "dataset\\five",
        "dataset\\four",
        "dataset\\go",
        "dataset\\happy",
        "dataset\\house",
        "dataset\\left",
        "dataset\\marvin",
        "dataset\\nine",
        "dataset\\no",
        "dataset\\off",
        "dataset\\on",
        "dataset\\one",
        "dataset\\right",
        "dataset\\seven",
        "dataset\\sheila",
        "dataset\\six",
        "dataset\\stop",
        "dataset\\three",
        "dataset\\tree",
        "dataset\\two",
        "dataset\\up",
        "dataset\\wow",
        "dataset\\yes",
        "dataset\\zero",
        "dataset\\_background_noise_"
    ]
    instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficients)


        # convert 2d MFCCs array into 4d array ->(# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.6, 0.1, ...] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service.instance is None:
        _Keyword_Spotting_Service.instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service.instance


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict("test/down.wav")
    keyword2 = kss.predict("test/left.wav")
    keyword3 = kss.predict("test/nine.wav")

    print(f"Predicted keywords: {keyword1}, {keyword2}, {keyword3}")

