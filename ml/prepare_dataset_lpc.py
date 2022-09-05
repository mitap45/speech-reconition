import librosa
import os
import json
import warnings
from scipy.signal import lfilter, resample
from scipy.signal.windows import hann
import numpy as np
from math import floor
from lpc.lpc import lpc_encode



warnings.filterwarnings("ignore")

DATASET_PATH = "dataset_turkish"
JSON_PATH = "data_turkish_lpc.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec

def prepare_dataset(dataset_path, json_path, order=13, hop_length=512, n_fft=2048):

    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "LPCs": [],
        "files": []

    }

    # loop through all the sub dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # we need to ensure that we're not at root level
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split("/")[-1] # dataset/down -> [dataset, down]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:

                # get the file path
                file_path = os.path.join(dirpath, f)

                # load the audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # enforce 1 sec. long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

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
                    LPCs=np.concatenate((A, G))


                    # store data
                    data["labels"].append(i-1)
                    data["LPCs"].append(LPCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path} : {i-1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":

    prepare_dataset(DATASET_PATH, JSON_PATH)