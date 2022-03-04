import os
import json
import random
import helpers


def remove_dot_in_last_sentence(text):
    if len(text) > 0 and text[-1] == "." :
        return text[:-1]
    return text

def format_data(data):
    fmt_data = []
    for d in data :
        fmt_data.append(json.loads(d))
    return fmt_data

def process():

    data = []
    data_dir = "/workspace/ASRDebugger/data/ST-AEDS/wav/"

    file = open("text.txt")
    for line in file.readlines():
        wav_path = line.split()[0]
        text = " ".join(line.split()[1:])

        wav_path = os.path.join(data_dir, wav_path)

        if os.path.exists(wav_path) :
            text = remove_dot_in_last_sentence(helpers.remove_punctuation(text))
            if bool(text):
                data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

    file.close()
    
    random.seed(123456)
    random.shuffle(data)
    
    os.makedirs("./manifests/", exist_ok=True)
    filepath = "./manifests/all.json"
    helpers.write_json_data(filepath, data)

    n = len(data)

    selection = ("selection", int(n * 0.7))
    seed = ("seed", int(n * 0.1))
    dev = ("dev", int(n * 0.1))
    test_size = n - selection[1] - seed[1] - dev[1]
    test = ("test", test_size)

    lower = 0
    
    for name, interval in [seed, dev, selection, test]:
        upper = lower + interval
        
        curr_data = data[lower:upper] 
        helpers.write_json_data(f"./manifests/{name}.json", curr_data)
        
        lower = upper

    helpers.write_json_data(f"./manifests/seed_plus_dev.json", data[0:(seed[1]+dev[1])])

def process_tts():

    data = []
    data_dir = "/workspace/ASRDebugger/data/ST-AEDS/TTS/"

    file = open("text.txt")
    for line in file.readlines():
        wav_path = line.split()[0]
        text = " ".join(line.split()[1:])

        wav_path = os.path.join(data_dir, wav_path)

        if os.path.exists(wav_path) and os.path.exists("/workspace/ASRDebugger/data/ST-AEDS/wav/"+line.split()[0]):
            text = remove_dot_in_last_sentence(helpers.remove_punctuation(text))
            if bool(text):
                data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

    file.close()
    
    random.seed(123456)
    random.shuffle(data)
    
    os.makedirs("./manifests/", exist_ok=True)
    filepath = "./manifests/all_tts.json"
    helpers.write_json_data(filepath, data)

    n = len(data)

    selection = ("selection", int(n * 0.7))
    seed = ("seed", int(n * 0.1))
    dev = ("dev", int(n * 0.1))
    test_size = n - selection[1] - seed[1] - dev[1]
    test = ("test", test_size)

    lower = 0
    
    for name, interval in [seed, dev, selection, test]:
        upper = lower + interval
        
        curr_data = data[lower:upper] 
        helpers.write_json_data(f"./manifests/{name}_tts.json", curr_data)
        
        lower = upper

    helpers.write_json_data(f"./manifests/seed_plus_dev_tts.json", data[0:(seed[1]+dev[1])])

def sample():
    seeds = [1, 2, 3]
    numbers = [50, 75, 100, 150, 200, 300, 400, 500]

    # load the seed dataset
    seed_data_path = f"./manifests/seed.json"
    # notice: seed data is REAL audio
    file = open(seed_data_path)
    seed_instances = file.readlines() 
    file.close()

    selection_json_fpath = f"./manifests/selection.json"

    file = open(selection_json_fpath)
    instances = file.readlines() 
    file.close()

    for seed in seeds:
        random.seed(seed)
        data = random.sample(instances, len(instances))
        
        for number in numbers :
            sample_data = data[:number]
            sample_data = seed_instances + sample_data
            sample_data = format_data(sample_data)
            folder_dir = f"./manifests/train/random/{number}/seed_{seed}/"
            
            os.makedirs(folder_dir, exist_ok=True)
            
            filepath = folder_dir + "train.json"

            helpers.write_json_data(filepath, sample_data)

def sample_tts():
    seeds = [1, 2, 3]
    numbers = [50, 75, 100, 150, 200, 300, 400, 500]

    # load the seed dataset
    seed_data_path = f"./manifests/seed.json"
    # notice: seed data is REAL audio
    file = open(seed_data_path)
    seed_instances = file.readlines() 
    file.close()

    selection_json_fpath = f"./manifests/selection_tts.json"

    file = open(selection_json_fpath)
    instances = file.readlines() 
    file.close()

    for seed in seeds:
        random.seed(seed)
        data = random.sample(instances, len(instances))
        
        for number in numbers :
            sample_data = data[:number]
            sample_data = seed_instances + sample_data
            sample_data = format_data(sample_data)
            folder_dir = f"./manifests/train/random_tts/{number}/seed_{seed}/"
            
            os.makedirs(folder_dir, exist_ok=True)
            
            filepath = folder_dir + "train.json"

            helpers.write_json_data(filepath, sample_data)

if __name__ == "__main__":

    process() # process original audio
    process_tts()

    sample() # sample original audio
    sample_tts()         