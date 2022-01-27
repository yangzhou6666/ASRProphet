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

import torch
import soundfile as sf
import time, gc

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2CTCTokenizer

import helpers


def parse_args():
    parser = argparse.ArgumentParser(description='Deepspeech')

    parser.register("type", "bool", lambda x: x.lower()
                    in ("yes", "true", "t", "1"))

    parser.add_argument("--val_manifest", type=str, required=True, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--model_name", type=str, required=True, help='path to the model pbmm')
    parser.add_argument("--vocab", type=str, required=True,
                        help='saves vocab.json for tokenizer')
    parser.add_argument("--model_tag", type=str, required=True, help='original model or fine-tuned model')
    parser.add_argument("--output_file", default="out.txt", type=str)
    parser.add_argument("--args.", type=str, help='indicating the tag version of the model (e.g. deepspeech, finetuned_deepspeech)')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--overwrite", action='store_true', help='overwrite the previous transcritiption')
    return parser.parse_args()


def main(args):

    if args.vocab == "facebook/wav2vec2-base-960h":
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(args.vocab)
    else :
        tokenizer = Wav2Vec2CTCTokenizer(args.vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)

    def wav2vec2_recognize_audio(audio_fpath):
        audio_input, _ = sf.read(audio_fpath)

        # transcribe
        input_values = tokenizer(audio_input, return_tensors="pt").input_values
        # input_values = input_values.to(self.device)

        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

        del audio_input, input_values, logits, predicted_ids
        torch.cuda.empty_cache()
        gc.collect()

        print("Wav2Vec2 transcription: ", transcription)

        return transcription


    random.seed(args.seed)
    np.random.seed(args.seed)

    references = []
    predictions = []

    file = open(args.val_manifest)

    for line in file.readlines():
        js_instance = json.loads(line)

        wav_path = js_instance["audio_filepath"]
        transcription_path = wav_path[:-3] + \
            args.model_tag + ".transcription.txt"

        if args.overwrite or (not os.path.exists(transcription_path)) or helpers.is_empty_file(transcription_path):
            print("Processing: ", transcription_path)

            transcription = wav2vec2_recognize_audio(wav_path)

            tfile = open(transcription_path, "w+")
            tfile.write("%s\n" % transcription)
            tfile.close()
        else:
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
