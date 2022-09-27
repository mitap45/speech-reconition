import librosa
import os
import json
import warnings
from plp import rastaplp

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset_turkish"
JSON_PATH = "data_turkish_plp.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec

def prepare_dataset(dataset_path, json_path):

    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "PLPs": [],
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

                    # extract the PLPs
                    PLPs = rastaplp(signal)

                    # store data
                    data["labels"].append(i-1)
                    data["PLPs"].append(PLPs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path} : {i-1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)