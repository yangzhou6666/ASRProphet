import os
import sys
import json
sys.path.append("..")
from scipy.special import softmax
from quartznet_asr.metrics import __levenshtein, word_error_rate
from power import Levenshtein, ExpandedAlignment
from power.aligner import PowerAligner
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import re
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Train Word Error Predictor')
    parser.add_argument("--seed_json_file", type=str,
                        required=True, help='path to seed json file')
    parser.add_argument("--selection_json_file", type=str,
                        required=True, help='path to input file')
    parser.add_argument("--random_json_file", type=str,
                        required=True, help='path to input file')
    parser.add_argument("--finetuned_ckpt", default=None,
                        type=str, help='path to finetuned ckpt')
    parser.add_argument("--log_dir", type=str, required=True,
                        help='saves logs in this directory')
    parser.add_argument("--num_sample", default=50, type=int,
                        help='number of selected samples')
    parser.add_argument("--sampling", default="", type=str,
                        help='sampling methods')
    parser.add_argument("--seed", default=1, type=int, help='seed id')
    parser.add_argument("--output_json_path", type=str,
                        required=True, help='json fpath to save the ranked texts')
    args=parser.parse_args()
    return args


def format_data(json_fpath):
    file = open(json_fpath)
    data = file.readlines()
    file.close()

    texts = []
    fpaths = []
    durations = []
    for d in data:
        js = json.loads(d)
        texts.append(js["text"])
        fpaths.append(js["audio_filepath"])
        durations.append(js["duration"])

    return texts, fpaths, durations


def sort(texts, fpaths, probabilities, durations):
    list1 = probabilities
    indexs = [i for i in range(len(probabilities))]
    list1, indexs = zip(*sorted(zip(list1, indexs)))

    indexs = list(indexs)[::-1]  # reverse

    texts = list(map(texts.__getitem__, indexs))
    fpaths = list(map(fpaths.__getitem__, indexs))
    probabilities = list(map(probabilities.__getitem__, indexs))
    durations = list(map(durations.__getitem__, indexs))

    return texts, fpaths, probabilities, durations

def dump_samples(samples, filename):
    output_dir = os.path.split(filename)[0]
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(sample)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class WordErrorSampler():
    '''
    The class for selecting test cases using the word-level error predictor
    '''
    def __init__(self, finetuned_ckpt, selection_json_file) -> None:
        '''load required parameters'''

        # path to the finetuned model
        self.finetuned_ckpt = finetuned_ckpt
        # path to the selection json file (the json file that contains the test cases)
        self.selection_json_file = selection_json_file

    def load_model(self):
        '''Load tokenizer and model'''

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_ckpt)
        # load the model
        self.model = AutoModelForTokenClassification.from_pretrained(self.finetuned_ckpt)
        self.model.cuda()
        self.model.eval()

    def load_sampling_data(self):
        '''Load the data for sampling'''

        # load the selection data
        self.selection_texts, self.selection_fpaths, self.selection_durations = format_data(self.selection_json_file)

        # tokenize the selection data
        test_encodings = self.tokenizer(self.selection_texts, truncation=True)
        test_dataset = TextDataset(test_encodings)

        # prepare data loader
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False)

        self.res = []
        # score for each test case
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            outputs = self.model(
                input_ids, attention_mask)

            probs = F.softmax(outputs.logits, -1)
            error_probs = probs[:, :, 1]
            self.res.append(error_probs[0].cpu().detach().numpy().tolist())
    
    def normal_sample(self, required_duration: float):
        '''select the test cases based on sum of their error score'''
        score = [sum(n)/len(n) for n in self.res]

        test_texts, fpaths, score, durations = sort(self.selection_texts, self.selection_fpaths, score, self.selection_durations)
        samples = []
        selected_duration = 0.0

        for text, fpath, duration in zip(test_texts, fpaths, durations):
            samples.append(json.dumps({"text": text, "audio_filepath": fpath, "duration": duration}) + "\n")
            selected_duration += json.loads(samples[-1].strip())['duration']
            if selected_duration >= required_duration:
                break

        print('sampled {} samples...'.format(len(samples)))

        return samples

    def sample_with_phone_enhancing(self, duration: float):
        pass

    def sample_with_word_enhancing(self, duration: float):
        pass



def main(args):
    '''
    Function to train and save the model.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seed_samples = [line for line in open(args.seed_json_file)]
    random_samples_duration = sum([json.loads(line.strip())['duration'] for line in open(args.random_json_file)])
    seed_duration = sum([json.loads(line.strip())['duration'] for line in open(args.seed_json_file)])
    required_duration = random_samples_duration - seed_duration
    assert required_duration > 0
    print('loading data....')

    err_sampler = WordErrorSampler(args.finetuned_ckpt, args.selection_json_file)
    err_sampler.load_model()
    err_sampler.load_sampling_data()
    samples = err_sampler.normal_sample(required_duration)

    # dump the samples
    output_json_file = os.path.join(args.output_json_path, str(
        args.num_sample), args.sampling, 'seed_' + str(args.seed), 'train.json')

    dump_samples(seed_samples + samples, output_json_file)


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_ckpt)
    main(args)

    

