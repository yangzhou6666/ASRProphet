import os,sys,json,random,argparse
from math import ceil
from typing import List
import numpy as np
from operator import itemgetter
from tqdm import tqdm
from g2p_en import G2p
get_phoneme_seq = G2p()

## add this codes so that `get_result.py` can import ErrorModelSampler from the main path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from text import _clean_text
import string
import pickle

from joblib import Parallel, delayed
from functools import partial

import multiprocessing
from multiprocessing import Pool, Manager

from helpers import print_dict

random_seed=42
random.seed(random_seed)
np.random.seed(random_seed)

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

def normalize_string(s, labels=labels, table=table, **unused_kwargs):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'

    Args:
        s: string to normalize
        labels: labels used during model training.

    Returns:
            Normalized string
    """

    def good_token(token, labels):
        s = set(labels)
        for t in token:
            if not t in s:
                return False
        return True

    try:
        text = _clean_text(s, ["english_cleaners"], table).strip()
        return ''.join([t for t in text if good_token(t, labels=labels)])
    except:
        print("WARNING: Normalizing {} failed".format(s))
        return None

def normalized_json_transcript(json_str):
  transcript = json.loads(json_str.strip())["text"]
  output = normalize_string(transcript,labels,table)
  return output


class ErrorModelSampler():
  def __init__(self, json_file, sampling_method, error_model_weights=None, verbose=True):
    self.json_file = json_file
    self.sampling_method = sampling_method
    
    
    if verbose: print('\tparsing json...')
    self.sentences = [normalized_json_transcript(line) for line in open(self.json_file)]
    self.json_lines = [line for line in open(self.json_file)]
    self.phone_sentences = []
    
    if verbose: print('\tgenerating_vocab...')
    if sampling_method == "triphone_rich" :
      self.phone_vocab = self.get_triphone_vocab(self.sentences)
    else :
      self.phone_vocab = phone_vocab
    
    self.phone_to_id = dict([(phone,i) for i,phone in enumerate(self.phone_vocab)])
    
    if verbose: print('\tcomputing phone freq for each sentence...')
    if sampling_method == "triphone_rich" :
      self.sent_wise_phone_freqs = self.get_sentence_wise_triphone_freqs()
    else :
      self.sent_wise_phone_freqs = self.get_sentence_wise_phone_freqs()
    
    self.phone_freqs = np.sum(np.array(self.sent_wise_phone_freqs),0)
    self.error_model_weights = error_model_weights
    self.acc_phone_freqs = np.zeros(len(self.phone_vocab))
    
    if sampling_method == "triphone_rich" :
      self.ideal_triphone_dists = np.ones(len(self.phone_vocab))/len(self.phone_vocab)
    else :
      self.ideal_phone_dists = np.ones(len(self.phone_vocab))/len(self.phone_vocab)
    
  def get_triphones(self, text:str)->List[List[str]]:
    """get triphone sequence for a given text

    Args:
        text (str): input text

    Returns:
        List[List[str]]: list of triphones in text
    """
    phones = self.get_phonemes(text)
    triphones = [phones[i:i+3] for i in range(len(phones)-2)]
    return triphones

  def format_list(self, l: List[str]):
      return "_".join(l)

  def get_triphone_vocab(self, sentences:List[str]) -> List[str]:
    """get triphone vocabulary

    Args:
        sentences (List[str]): corpus of sentences

    Returns:
        List[str]: unique list of triphones vocabularies
    """
    phone_vocab = []
    for text in sentences:
      triphones = self.get_triphones(text)
      triphones = [self.format_list(triphone) for triphone in triphones]
      phone_vocab += triphones
    phone_vocab = np.unique(phone_vocab).tolist()
    phone_vocab += [MASK, ' ']
    return phone_vocab

  def get_phonemes(self, text):
    phone_list = get_phoneme_seq(text)
    phone_list = [item if len(item)!=3 else item[:-1] for item in phone_list]
    phone_list = [item for item in phone_list if item not in {"'"}]
    return phone_list

  def get_sentence_wise_triphone_freqs(self):
    sent_wise_triphone_freqs = []
    for text in self.sentences:
      freq = np.zeros(len(self.phone_vocab))
      triphones = self.get_triphones(text)
      triphones = [self.format_list(triphone) for triphone in triphones]
      for phone in triphones:
        freq[self.phone_to_id[phone]] +=1
      self.phone_sentences.append(triphones)
      sent_wise_triphone_freqs.append(freq)
    return sent_wise_triphone_freqs

  
  def get_sentence_wise_phone_freqs(self):
    sent_wise_phone_freqs = []
    for text in self.sentences:
      freq = np.zeros(len(self.phone_vocab))
      phones = self.get_phonemes(text)
      for phone in phones:
        freq[self.phone_to_id[phone]] +=1
      self.phone_sentences.append(phones)
      sent_wise_phone_freqs.append(freq)
    return sent_wise_phone_freqs

  def get_f(self,freq,tau=500):
    return 1 - np.exp(-freq/tau)


  def get_f2(self, freq):
    '''
    From A SUBMODULAR OPTIMIZATION APPROACH TO SENTENCE SET SELECTION
    sum log(f_i(S))
    '''
    return np.log(freq + 1)

  def compute_euclidean_distance(self, v_1:List[float], v_2: List[float]) -> float:
    """Compute the Euclidean distance between two vectors.

    Args:
        v_1 (List[float]): an array of float
        v_2 (List[float]): an array of float

    Returns:
        float: euclidean distance
    """
    return np.linalg.norm(np.array(v_1) - np.array(v_2))
  
  def get_proportion(self, l:List[float])->List[float]:
    s = np.sum(l)
    if s == 0 : s = 1
    return l/s
    

  def select_text_and_update_phone_freq(self, sampling_method, weight_id):
    min_indices = []
    min_sentences = []
    max_score = -1e10 

    for i,sentence in enumerate(self.sentences):
      
      if sampling_method == 'diversity_enhancing' :
        f_i = self.get_f(self.acc_phone_freqs)
        f_f = self.get_f(self.acc_phone_freqs + self.sent_wise_phone_freqs[i])
        score = (f_f - f_i) * self.error_model_weights[weight_id][i]
      elif sampling_method == 'without_diversity_enhancing':
        score = self.error_model_weights[weight_id][i]
      elif sampling_method == 'pure_diversity':
        # print("acc_phone_freqs\n", self.acc_phone_freqs)
        # print("sent_wise_phone_freqs[i]\n", self.sent_wise_phone_freqs[i])
        f_i = self.get_f2(self.acc_phone_freqs)
        f_f = self.get_f2(self.acc_phone_freqs + self.sent_wise_phone_freqs[i])
        score = (f_f - f_i)
      elif sampling_method == 'triphone_rich':
        '''
        dist = self.compute_trip_distrbution(xxx + text_to_be_selected) # if we add this text to the selected texts
        assert len(dist) == triphone_size
        idea_dist = [1.0 / triphone_size for _ in range(triphone_size)]
        score = self.compute_e_distance()
        '''
        new_dists = self.acc_phone_freqs + self.sent_wise_phone_freqs[i]
        assert len(self.ideal_triphone_dists) == len(new_dists)
        input_feature = self.get_proportion(new_dists) 
      else :
        raise ValueError('sampling_method {} not supported'.format(sampling_method))
      
      if sampling_method == 'triphone_rich':
        # give negative score because we want to choose the score with lowest value
        score = -self.compute_euclidean_distance(input_feature, self.ideal_triphone_dists)
      else :
        score = score[4:-2]
        score = np.sum(score)/len(self.phone_sentences[i])
      
      if score > max_score:
        max_score = score
        max_indices = [i]
        max_sentences = [sentence]
      elif score == max_score:
        max_indices.append(i)
        max_sentences.append(sentence)

    selected_sentence_idx = random.choice(max_indices)
    selected_sentence = self.sentences[selected_sentence_idx]
    self.acc_phone_freqs += self.sent_wise_phone_freqs[selected_sentence_idx]
    selected_json_line = self.json_lines[selected_sentence_idx]
    self.sentences.pop(selected_sentence_idx)
    self.sent_wise_phone_freqs.pop(selected_sentence_idx)
    self.json_lines.pop(selected_sentence_idx)
    self.phone_sentences.pop(selected_sentence_idx)
    _ = [self.error_model_weights[w_idx].pop(selected_sentence_idx) 
         for w_idx in range(len(self.error_model_weights))]
    return selected_json_line

  def sample(self,duration):
    samples = []
    selected_duration = 0.0
    total_sents = len(self.sentences)
    for i in range(total_sents):
      weight_id = i%len(self.error_model_weights)
      
      samples.append(self.select_text_and_update_phone_freq(self.sampling_method, weight_id))  
      
      selected_duration += json.loads(samples[-1].strip())['duration']
      if selected_duration >= duration:
        break
    print('sampled {} samples...'.format(len(samples)))
    return samples

def dump_samples(samples,filename):
  output_dir = os.path.split(filename)[0]
  os.makedirs(output_dir,exist_ok=True)
  with open(filename,'w') as f:
    for sample in samples:
      f.write(sample)

def get_samples(sampler,duration):
  return sampler.sample(duration)

def get_json_duration(json_file):
  return sum([json.loads(line.strip())['duration'] for line in open(json_file)])

def parse_args():
  parser = argparse.ArgumentParser(description='error model sampling')
  parser.add_argument("--selection_json_file", type=str, help='path to json file from where sentences are selected')
  parser.add_argument("--sampling_method", type=str, default='diversity_enhancing', help='sampling method')
  parser.add_argument("--seed_json_file", type=str, help='path to json file containing seed sentences')
  parser.add_argument("--error_model_weights", type=str, help='weights provided by error model inference')
  parser.add_argument("--random_json_path",type=str,
    help='path to dir containing json files for randomly selected sentences, used to ensure same amount of speech time')
  parser.add_argument("--output_json_path", type=str, 
    help='path to dir containing json files for sentences selected via error model weights')
  parser.add_argument("--exp_id", type=str, help='experiment id')
  args=parser.parse_args()
  return args

def main(args):
  selection_json_file = args.selection_json_file
  seed_json_file = args.seed_json_file
  random_json_path = args.random_json_path
  output_json_path = args.output_json_path
  seed_samples = [line for line in open(seed_json_file)]
  weights_file = args.error_model_weights
  exp_id = args.exp_id
  for num_samples in [100, 200, 300, 400]:
    weights = pickle.load(open(weights_file,'rb'))
    weights_list = [weights]  
    random_json_file = os.path.join(random_json_path,str(num_samples),'seed_'+exp_id,'train.json')
    random_samples_duration = get_json_duration(random_json_file)
    seed_duration = get_json_duration(seed_json_file)
    required_duration = random_samples_duration - seed_duration
    assert required_duration > 0
    output_json_file = os.path.join(output_json_path,str(num_samples),'seed_'+exp_id,'train.json')
    sampler = ErrorModelSampler(selection_json_file, args.sampling_method, error_model_weights=weights_list)
    samples = get_samples(sampler, required_duration)
    dump_samples(seed_samples + samples,output_json_file)

    output_json_file = os.path.join(output_json_path,str(num_samples),'seed_'+exp_id,'train_no_seed.json')
    dump_samples(samples,output_json_file)

if __name__=="__main__":
  args = parse_args()
  print_dict(vars(args))
  print()
  main(args)