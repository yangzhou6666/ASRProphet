import argparse
import itertools
from typing import List
from tqdm import tqdm
import math
import toml
import random
import numpy as np
import pickle
import time
import os
import subprocess
import json

 
import helpers


def parse_args():
    parser = argparse.ArgumentParser(description='Deepspeech')

    parser.register("type", "bool", lambda x: x.lower() in ("yes", "true", "t", "1"))

    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--model", type=str, help='path to the model pbmm')
    parser.add_argument("--scorer", default=None, type=str, required=True, help='path to the model scorer')
    parser.add_argument("--model_tag", type=str, help='indicating the tag version of the model (e.g. deepspeech, finetuned_deepspeech)')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--output_file",default="out.txt",type=str)
    return parser.parse_args()


def deepspeech_recognize_audio(model_path, scorer_path, audio_fpath):
    cmd = f"deepspeech --model {model_path}" + \
        f" --scorer {scorer_path}" + \
        f" --audio {audio_fpath}"

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()

    transcription = out.decode("utf-8")[:-1]

    print("DeepSpeech transcription: %s" % transcription)
    return transcription


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_tag = "deepspeech"


    references = []
    predictions = []

    file = open(args.val_manifest)
    
    for line in file.readlines():
        js_instance = json.loads(line)

        wav_path = js_instance["audio_filepath"]
        transcription_path = wav_path[:-3] + model_tag + ".transcription.txt"


        if (not os.path.exists(transcription_path)) or helpers.is_empty_file(transcription_path):
            print("Processing: ", transcription_path)

            transcription = deepspeech_recognize_audio(args.model, args.scorer, wav_path)

            tfile = open(transcription_path, "w+")
            tfile.write("%s\n" % transcription)
            tfile.close()
        else :
            tfile = open(transcription_path)
            transcription = tfile.readline()[:-1]
            tfile.close()
        
        references.append(helpers.preprocess_text(js_instance["text"]))
        predictions.append(helpers.preprocess_text(transcription))

    file.close()
    
    if args.output_file:
        helpers.print_sentence_wise_wer(
            predictions, references, args.output_file, args.val_manifest)


if __name__ == "__main__":
    args = parse_args()

    helpers.print_dict(vars(args))

    main(args)
