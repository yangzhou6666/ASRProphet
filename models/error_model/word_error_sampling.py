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
import string
from tqdm import tqdm
import torch.nn.functional as F
from g2p_en import G2p

labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
punctuation = string.punctuation
punctuation = punctuation.replace("+", "")
punctuation = punctuation.replace("&", "")
for l in labels:
    punctuation = punctuation.replace(l, "")
table = str.maketrans(punctuation, " " * len(punctuation))


MASK = '<mask>'
SOS = '<s>'
EOS = '</s>'


get_phoneme_seq = G2p()
phone_vocab = get_phoneme_seq.phonemes[0:4]
phone_vocab += np.unique([item if len(item)!=3 else item[:-1] for item in get_phoneme_seq.phonemes[4:]]).tolist()
phone_vocab += [MASK, ' ']


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

def get_f(freq,tau=500):
    return 1 - np.exp(-freq/tau)

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

        # rank the test cases based on the score
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

    def sample_with_phone_enhancing(self, required_duration: float):
        '''select the test cases based on sum of their error score and consider the phone diversisty enhancing'''

        # we cannot pre-compute a static metric for all texts in advance
        # we need to compute it on the fly

        # rank the test cases based on the score
        score = [sum(n)/len(n) for n in self.res]
        test_texts, fpaths, score, durations = sort(self.selection_texts, self.selection_fpaths, score, self.selection_durations)
        samples = []
        selected_duration = 0.0

        path_score = {}
        for text, fpath, this_score in zip(test_texts, fpaths, score):
            path_score[fpath] = this_score


        while (selected_duration < required_duration and len(test_texts) > 0):
            # first, start with the case with the highest score
            text = test_texts.pop(0)
            fpath = fpaths.pop(0)
            this_score = score.pop(0)
            duration = durations.pop(0)

            selected_duration += duration

            samples.append(json.dumps({"text": text, "audio_filepath": fpath, "duration": duration}) + "\n")

            # after selecting a test case, we need to update the phoneme frequency
            self.acc_phone_freqs += self.sent_wise_phone_freqs[fpath]['freq']

            # update the score for the remaining test cases
            new_scores = []
            for i in range(len(score)):
                new_path = fpaths[i]
                f_old = get_f(self.acc_phone_freqs)
                f_new = get_f(self.acc_phone_freqs + self.sent_wise_phone_freqs[new_path]['freq'])
                new_score = (f_new - f_old) * path_score[new_path]
                new_score = np.sum(new_score)
                new_scores.append(new_score)
            
            test_texts, fpaths, score, durations = sort(test_texts, fpaths, new_scores, durations)

        return samples



    def sample_with_word_enhancing(self, duration: float):
        pass

    def get_phonemes(self, text):
        '''compute the phoneme frequency of a text'''
        phone_list = get_phoneme_seq(text)
        phone_list = [item if len(item)!=3 else item[:-1] for item in phone_list]
        phone_list = [item for item in phone_list if item not in {"'"}]
        return phone_list

    def get_sentence_wise_phone_freqs(self):
        '''compute the phoneme frequency of each sentence'''
        self.phone_vocab = phone_vocab
        self.acc_phone_freqs = np.zeros(len(self.phone_vocab))
        self.sent_wise_phone_freqs = {}
        self.phone_to_id = dict([(phone,i) for i,phone in enumerate(self.phone_vocab)])
        for text, path in zip(self.selection_texts, self.selection_fpaths):
            freq = np.zeros(len(self.phone_vocab))
            phones = self.get_phonemes(text)
            for phone in phones:
                freq[self.phone_to_id[phone]] +=1
            self.sent_wise_phone_freqs[path] = {'freq': freq, 'phones': phones, 'text': text}

        return self.sent_wise_phone_freqs


def main(args):
    '''
    Function to train and save the model.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # compute the required duration to be sampled
    seed_samples = [line for line in open(args.seed_json_file)]
    random_samples_duration = sum([json.loads(line.strip())['duration'] for line in open(args.random_json_file)])
    seed_duration = sum([json.loads(line.strip())['duration'] for line in open(args.seed_json_file)])
    required_duration = random_samples_duration - seed_duration
    assert required_duration > 0
    print('loading data....')

    err_sampler = WordErrorSampler(args.finetuned_ckpt, args.selection_json_file)
    err_sampler.load_model()
    err_sampler.load_sampling_data()
    err_sampler.get_sentence_wise_phone_freqs()
    samples = err_sampler.sample_with_phone_enhancing(required_duration)

    # dump the samples

    exit()
    output_json_file = os.path.join(args.output_json_path, str(
        args.num_sample), args.sampling, 'seed_' + str(args.seed), 'train.json')

    dump_samples(seed_samples + samples, output_json_file)


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_ckpt)
    main(args)

    

