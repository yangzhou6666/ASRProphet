import os
import json
import random
import helpers

def format_data(data):
    fmt_data = []
    for d in data :
        fmt_data.append(json.loads(d))
    return fmt_data


def process(path_to_l2arctic_release_v5):
    data = []



    for name in os.listdir(path_to_l2arctic_release_v5):
        root_dir = f"{path_to_l2arctic_release_v5}{name}/"
        # Path to each original sub-dataset
        transcript_dir = os.path.join(root_dir, "transcript")
        wav_dir = os.path.join(root_dir, "wav")

        for filename in os.listdir(transcript_dir):
            file_id = filename.split('.')[0]

            # get text (transcript)
            with open(os.path.join(transcript_dir, file_id + '.txt')) as f:
                text = f.readlines()[0]

            wav_path = os.path.join(wav_dir, file_id + '.wav')
            data.append({"text": text, "audio_filepath": wav_path, "duration": helpers.measure_audio_duration(wav_path)})

    
        random.seed(123456)
        random.shuffle(data)

        path_to_store = os.path.join("./processed", name, 'manifests')
        
        os.makedirs(path_to_store, exist_ok=True)
        helpers.write_json_data(os.path.join(path_to_store, "all.json"), data)

        n = len(data)

        selection = ("selection", int(n * 0.75))
        seed = ("seed", int(n * 0.1))
        dev = ("dev", int(n * 0.1))
        test_size = n - selection[1] - seed[1] - dev[1]
        test = ("test", test_size)

        lower = 0
        
        for name, interval in [seed, dev, selection, test]:
            upper = lower + interval
            
            curr_data = data[lower:upper] 
            helpers.write_json_data(f"{path_to_store}/{name}.json", curr_data)
            
            lower = upper

        helpers.write_json_data(f"{path_to_store}/seed_plus_dev.json", data[0:(seed[1]+dev[1])])



def sample(path_to_l2arctic_release_v5):
    seeds = [1, 2, 3]
    numbers = [50, 100, 200, 500]

    ## load selection.json
    for name in os.listdir(path_to_l2arctic_release_v5):
        selection_json_fpath = f"./processed/{name}/manifests/selection.json"
    
        file = open(selection_json_fpath)
        instances = file.readlines() 
        file.close()
        ## random select for several seeds
        for seed in seeds :
            random.seed(seed)
            data = random.sample(instances, len(instances))
            
            for number in numbers :
                sample_data = data[:number]
                sample_data = format_data(sample_data)
                folder_dir = f"./processed/{name}/manifests/train/random/{number}/seed_{seed}/"
                
                os.makedirs(folder_dir, exist_ok=True)
                
                filepath = folder_dir + "train.json"
                
                ## save to external files 
                helpers.write_json_data(filepath, sample_data)

if __name__ == "__main__":
    path_to_l2arctic_release_v5 = "./l2arctic_release_v5/"
    # Path to the original dataset


    ## Preprocess
    process(path_to_l2arctic_release_v5)

    ## Sample
    sample(path_to_l2arctic_release_v5)

