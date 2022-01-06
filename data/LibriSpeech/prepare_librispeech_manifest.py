import os
import glob
import json
import random


def idx_to_file(idx):
    return "/".join(idx.split("-")[:-1])


if __name__ == "__main__":

    DATA_TYPES = ["test-clean", "test-other", "dev-clean", "dev-other"]

    data = []

    for data_type in DATA_TYPES:

        data_dir = f"{data_type}/"

        ## root_dir needs a trailing slash (i.e. /root/dir/)
        for filename in sorted(glob.iglob(data_dir + '**/*.trans.txt', recursive=True)):

            file = open(filename)
            for line in file.readlines():
                idx = line.split()[0]
                text = " ".join(line.split()[1:])

                fname = os.path.join(data_dir, idx_to_file(idx), idx)
                flac_path = fname + ".flac"
                wav_path = fname + ".wav"

                wav_path = f"data/LibriSpeech/{wav_path}"

                data.append({"text": text, "audio_filepath": wav_path})

            file.close()

    
    random.seed(123456)
    random.shuffle(data)
    
    os.makedirs("manifests/", exist_ok=True)

    with open("manifests/all.json", 'w') as f:
        for d in data :
            json.dump(d, f)
            f.write("\n")
