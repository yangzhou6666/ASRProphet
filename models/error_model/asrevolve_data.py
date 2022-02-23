import torch
import numpy as np
from g2p_en import G2p
from math import floor
import random

from tqdm import tqdm
from joblib import Parallel, delayed

from helpers import normalized_json_transcript
from seq2edits_utils import ndiff
from power.aligner import PowerAligner

import multiprocessing
from multiprocessing import Pool, Manager

SUCCESSFUL_TEST_CASE = 0
FAILED_TEST_CASE = 1

def get_WER_from_para(para):
  return float(para.strip().split('\n')[2][5:])

def generate_asrevolve_training_data_from_hypotheses_file(path_hypotheses, skip_zero_CER=False):
  with open(path_hypotheses) as f:
    paragraphs = f.read().strip().split('\n\n')
    ref_hyp_pairs = [(para.strip().split('\n')[3][5:], para.strip().split('\n')[4][5:]) \
                      for para in paragraphs if (not skip_zero_CER or get_WER_from_para(para)!=0.0)]
    pool = Pool(16)
    multiprocessed_output = list(filter(None, pool.map(__generate_error_label, tqdm(ref_hyp_pairs, total=len(ref_hyp_pairs)))))
    pool.close()
    error_labels, reference_texts = zip(*multiprocessed_output)
    print(error_labels[0])
    print(reference_texts[0])
    # return list(zip(error_labels, reference_texts))
    return error_labels, reference_texts

def __generate_error_label(reference_hypothesis_pair):
    reference, hypothesis = reference_hypothesis_pair
    # reference: the ground truth text
    # hypothesis: the transcribed text

    label = None

    if reference == hypothesis :
      label = SUCCESSFUL_TEST_CASE
    else :
      label = FAILED_TEST_CASE

    return label, reference


if __name__ == '__main__':

  ref = 'anatomy hi hello democracy'
  hyp = 'and that to me hello de mo gracy'

  error_label = __generate_error_label((ref,hyp))

  print(error_label[0])
