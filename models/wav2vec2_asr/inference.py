import os
import json
import torch
import random
import helpers
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoModelForCTC, Wav2Vec2Processor


def parse_args():
    parser = argparse.ArgumentParser(description='wav2vec2')

    parser.add_argument("--wav_dir", type=str, help='directory to the wav file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--output_file", default="out.txt", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--model", default="wav2vec", type=str)
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--batch_size", default=32, type=int, help='seed')
    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    models = {"wav2vec": "facebook/wav2vec2-base-960h",
              "hubert": "facebook/hubert-large-ls960-ft"}
    model = AutoModelForCTC.from_pretrained(models[args.model])
    feature_extractor = Wav2Vec2Processor.from_pretrained(models[args.model])
    if args.checkpoint:
        model = AutoModelForCTC.from_pretrained(args.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    references = []
    predictions = []

    def add_prefix(example):
        example['audio_filepath'] = os.path.join(args.wav_dir, example['audio_filepath'])
        return example
    
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio_filepath"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        ).input_values
        return inputs

    dataset = load_dataset("json", data_files={"validation": args.val_manifest}, split="validation")
    dataset = dataset.map(add_prefix)
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16_000))
    dataset = preprocess_function(dataset[::])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch.to(device)
            logits = model(input_ids).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = feature_extractor.batch_decode(predicted_ids)
            for trans in transcription:
                predictions.append(helpers.preprocess_text(trans))

    with open(args.val_manifest) as f:
        for line in f.readlines():
            js_instance = json.loads(line)
            references.append(helpers.preprocess_text(js_instance["text"]))

    if args.output_file:
        helpers.print_sentence_wise_wer(
            predictions, references, args.output_file, args.val_manifest)


if __name__ == "__main__":
    args = parse_args()

    helpers.print_dict(vars(args))

    main(args)
