import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import argparse
import random
from math import floor,ceil
from functools import partial
from shutil import copyfile
import sklearn

from model import ErrorClassifierPhoneBiLSTM_V2
from data import error_classifier_collate_fn, geneate_error_data_from_hypotheses_file
from asrevolve_data import generate_asrevolve_training_data_from_hypotheses_file
from metrics import xent_loss, error_classifier_errors, get_precision_recall_f1
from helpers import print_dict, warmup_decay_policy, save_model

from failure_estimator import HuggingFaceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description='asrevolve_error_model')
    parser.add_argument("--train_path", type=str, required=True, help='path to training data')
    parser.add_argument("--test_path", type=str, required=True, help='path to testing data')
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--log_dir", type=str, required=True, help='saves logs in this directory')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    args=parser.parse_args()
    return args


def randomize(texts, labels):
    assert len(texts) == len(labels)
    indexs = np.array(list(range(len(texts))))
    np.random.shuffle(indexs)
    indexs = list(indexs)
    ## get array items given a list of indexes
    texts = list(map(texts.__getitem__, indexs))
    labels = list(map(labels.__getitem__, indexs))
    return texts, labels

def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  print('loading data....')

  train_labels, train_texts = generate_asrevolve_training_data_from_hypotheses_file(args.train_path)

  test_labels, test_texts = generate_asrevolve_training_data_from_hypotheses_file(args.test_path)
  
  train_texts, train_labels = randomize(train_texts, train_labels)
  
  test_labels = list(test_labels)
  test_texts = list(test_texts)
#   train_size = floor(args.train_portion*len(labels))
#   train_labels = labels[0:train_size]
#   train_texts = texts[0:train_size]

#   dev_labels = labels[train_size:]
#   dev_texts = texts[train_size:]

  print('train_size: {} dev_size: {}'.format(len(train_labels),len(test_labels)))

  estimator = HuggingFaceTransformer(name="bert-base-uncased", output_dir=args.output_dir, logging_dir=args.log_dir)

  estimator.fit(train_texts, train_labels, test_texts, test_labels)

  probability = estimator.predict(test_texts)

  predictions = [(1) if prob > 0.5 else 0 for prob in probability]

  accuracy = sklearn.metrics.accuracy_score(test_labels, predictions)
  f1 = sklearn.metrics.f1_score(test_labels, predictions)

  print(f"accuracy: {accuracy}")
  print(f"f1 : {f1}")


if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)
