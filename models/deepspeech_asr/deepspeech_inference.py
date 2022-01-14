import argparse
from curses import flash
import itertools
import re
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

import multiprocessing as mp
 
import helpers


def parse_args():
    parser = argparse.ArgumentParser(description='Deepspeech')

    parser.register("type", "bool", lambda x: x.lower() in ("yes", "true", "t", "1"))

    parser.add_argument("--val_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--tts_manifest", type=str, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--model", type=str, help='path to the model pbmm')
    parser.add_argument("--scorer", default=None, type=str, required=True, help='path to the model scorer')
    parser.add_argument("--model_tag", type=str, help='indicating the tag version of the model (e.g. deepspeech, finetuned_deepspeech)')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--output_file",default="out.txt",type=str)
    return parser.parse_args()


def deepspeech_recognize_audio(model_path, scorer_path, audio_fpath):
    result = subprocess.run(['deepspeech', '--model', model_path, '--scorer', scorer_path, '--audio', audio_fpath], stdout=subprocess.PIPE)
    transcription = result.stdout.decode("utf-8")[:-1]

    print("DeepSpeech transcription: %s" % transcription)
    return transcription

def recognize_tts(model, scorer, wav_path, transcription_path):
    transcription = deepspeech_recognize_audio(model, scorer, wav_path)

    tfile = open(transcription_path, "w+")
    tfile.write("%s\n" % transcription)
    tfile.close()

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

    tts_preds = []
    transcription_paths = []

    pool = mp.Pool(mp.cpu_count())
    print(args.tts_manifest, flush=True)
    with open(args.tts_manifest) as f:
        for line in f.readlines():
            js_instance = json.loads(line)
            wav_path = js_instance["audio_filepath"]
            transcription_path = wav_path[:-3] + model_tag + ".transcription.txt"
            transcription_paths.append(transcription_path)
            # recognize_tts(args.model, args.scorer, wav_path, transcription_path)
            # exit()
            if (not os.path.exists(transcription_path)) or helpers.is_empty_file(transcription_path):
                pool.apply_async(recognize_tts, args=(args.model, args.scorer, wav_path, transcription_path))
        pool.close()
        pool.join()

    for transcription_path in tqdm(transcription_paths):
        tfile = open(transcription_path)
        transcription = tfile.readline()[:-1]
        tfile.close()
        
        tts_preds.append(helpers.preprocess_text(transcription))

    if args.output_file:
        helpers.print_sentence_wise_wer(
            predictions, references, tts_preds, args.output_file, args.val_manifest)


if __name__ == "__main__":
    args = parse_args()

    helpers.print_dict(vars(args))

    main(args)
