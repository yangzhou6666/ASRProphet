import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import argparse
import random
from math import floor, ceil
from functools import partial
from shutil import copyfile
import sklearn

from model import ErrorClassifierPhoneBiLSTM_V2
from data import error_classifier_collate_fn, geneate_error_data_from_hypotheses_file
from asrevolve_data import generate_asrevolve_training_data_from_hypotheses_file
from metrics import xent_loss, error_classifier_errors, get_precision_recall_f1
from helpers import print_dict, warmup_decay_policy, save_model

from failure_estimator import HuggingFaceTransformer

import json


def parse_args():
    parser = argparse.ArgumentParser(description='asrevolve_error_model')
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
    parser.add_argument("--exp_id", default=1, type=int, help='experiment id')
    parser.add_argument("--output_json_path", type=str,
                        required=True, help='json fpath to save the ranked texts')
    args = parser.parse_args()
    return args


def format_data(data):
    texts = []
    fpaths = []
    durations = []
    for d in data:
        js = json.loads(d)
        texts.append(js["text"])
        fpaths.append(js["audio_filepath"])
        durations.append(js["duration"])
    return texts, fpaths, durations


def load_json_data(json_fpath):
    file = open(json_fpath)
    data = file.readlines()
    file.close()
    return format_data(data)


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


def main(args):

    random.seed(args.exp_id)
    np.random.seed(args.exp_id)
    torch.manual_seed(args.exp_id)

    seed_samples = [line for line in open(args.seed_json_file)]
    random_samples_duration = sum([json.loads(line.strip())['duration'] for line in open(args.random_json_file)])
    seed_duration = sum([json.loads(line.strip())['duration'] for line in open(args.seed_json_file)])
    required_duration = random_samples_duration - seed_duration
    assert required_duration > 0
    print('loading data....')

    texts, fpaths, durations = load_json_data(args.selection_json_file)

    estimator = HuggingFaceTransformer(
        name=args.finetuned_ckpt, output_dir=args.finetuned_ckpt, logging_dir=args.log_dir)

    probabilities = estimator.predict(texts)

    texts, fpaths, probabilities, durations = sort(texts, fpaths, probabilities, durations)

    samples = []
    selected_duration = 0.0
    for text, fpath, duration in zip(texts, fpaths, durations):
        samples.append(json.dumps({"text": text, "audio_filepath": fpath, "duration": duration}) + "\n")
        selected_duration += json.loads(samples[-1].strip())['duration']
        if selected_duration >= required_duration:
            break
    print('sampled {} samples...'.format(len(samples)))

    output_json_file = os.path.join(args.output_json_path, str(
        args.num_sample), 'seed_' + str(args.exp_id), 'train.json')

    dump_samples(seed_samples + samples, output_json_file)


if __name__ == "__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)
