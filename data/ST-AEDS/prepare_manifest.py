import os
import glob
import json
import random
import helpers

def remove_dot_in_last_sentence(text):
    if len(text) > 0 and text[-1] == "." :
        return text[:-1]
    return text

if __name__ == "__main__":

    data = []

    data_dir = "data/"

    filename = "data/text.txt"

    file = open(filename)
    for line in file.readlines():
        wav_path = line.split()[0]
        text = " ".join(line.split()[1:])

        wav_path = os.path.join(data_dir, wav_path)

        wav_path = f"/workspace/ASRDebugger/data/ST-AEDS/{wav_path}"

        if os.path.exists(wav_path) :
            text = remove_dot_in_last_sentence(helpers.remove_punctuation(text))
            if bool(text) :
                data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

    file.close()

    # print("XXXX")
    # print(len(data))
    
    random.seed(123456)
    random.shuffle(data)
    
    os.makedirs("manifests/", exist_ok=True)
    filepath = "manifests/all.json"
    helpers.write_json_data(filepath, data)

    n = len(data)

    selection = ("selection", int(n * 0.7))
    seed = ("seed", int(n * 0.03))
    dev = ("dev", int(n * 0.03))
    test_size = n - selection[1] - seed[1] - dev[1]
    test = ("test", test_size)

    lower = 0
    
    for name, interval in [seed, dev, selection, test]:
        upper = lower + interval
        
        curr_data = data[lower:upper] 
        helpers.write_json_data(f"manifests/{name}.json", curr_data)
        
        lower = upper

    helpers.write_json_data(f"manifests/seed_plus_dev.json", data[0:(seed[1]+dev[1])])
