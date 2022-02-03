import os
import glob
import json
import random
import helpers


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

                wav_path = f"/workspace/ASRDebugger/data/LibriSpeech/{wav_path}"

                data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

            file.close()

    
    random.seed(123456)
    random.shuffle(data)
    
    os.makedirs("manifests/", exist_ok=True)
    filepath = "manifests/all.json"
    helpers.write_json_data(filepath, data)

    n = len(data)

    selection = ("selection", int(n * 0.4))
    seed = ("seed", int(n * 0.05))
    dev = ("dev", int(n * 0.05))
    test_size = n - selection[1] - seed[1] - dev[1]
    test = ("test", test_size)

    lower = 0
    
    for name, interval in [seed, dev, selection, test]:
        upper = lower + interval
        
        curr_data = data[lower:upper] 
        helpers.write_json_data(f"manifests/{name}.json", curr_data)
        
        lower = upper

    helpers.write_json_data(f"manifests/seed_plus_dev.json", data[0:(seed[1]+dev[1])])

