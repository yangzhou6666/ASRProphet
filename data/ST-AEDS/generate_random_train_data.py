import os
import json
import random
import helpers
from config import WORK_DIR


def format_data(data):
    fmt_data = []
    for d in data :
        fmt_data.append(json.loads(d))
    return fmt_data


if __name__ == "__main__" :

    ## load selection.json
    selection_json_fpath = f"{WORK_DIR}/manifests/selection.json"
    seeds = [1, 2, 3]
    numbers = [50, 100, 1500, 2000]
    
    file = open(selection_json_fpath)
    instances = file.readlines() 
    file.close()

    ## random select for several seeds
    for seed in seeds :
        random.seed(seed)
        data = random.sample(instances, len(instances))
        
        ## random select for vaious number of instances, 200, 400, 600
        for number in numbers :
            sample_data = data[:number]
            sample_data = format_data(sample_data)
            folder_dir = f"manifests/train/random/{number}/seed_{seed}/"
            
            os.makedirs(folder_dir, exist_ok=True)
            
            filepath = folder_dir + "train.json"
            
            ## save to external files 
            helpers.write_json_data(filepath, sample_data)
            
