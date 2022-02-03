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


MASK = '<mask>'
SOS = '<s>'
EOS = '</s>'


get_phoneme_seq = G2p()
phone_list = get_phoneme_seq.phonemes[0:4]
phone_list += np.unique([item if len(item)!=3 else item[:-1] for item in get_phoneme_seq.phonemes[4:]]).tolist()
phone_list += [MASK, ' ']

class Phonemes(object):
  # Vowels
  monophthongs = set(['AO', 'AA', 'IY', 'UW', 'EH', 'IH', 'UH', 'AH', 'AE']) #'AX',
  diphthongs = set(['EY', 'AY', 'OW', 'AW', 'OY'])
  # Add r-colored monophthongs? 
  r_vowels = set(['ER']) #'AXR'  # The remaining are split into two tokens.
  vowels = set.union(monophthongs, diphthongs, r_vowels)
  
  # Consonants
  c_stops = set(['P','B','T','D','K','G'])
  c_afficates = set(['CH','JH'])
  c_fricatives = set(['F','V','TH','DH','S','Z','SH','ZH','HH'])
  c_nasals = set(['M','N','NG']) #,'EM', ENG','EN'
  c_liquids = set(['L','R']) #'EL','DX','NX' 
  c_semivowels = set(['Y','W']) # 'Q'
  consonants = set.union(c_stops, c_afficates, c_fricatives, c_nasals, c_liquids, c_semivowels)

  vowels_id = 0
  consonants_id = 1

  monophthongs_id = 0
  diphthongs_id = 1
  r_vowels_id = 2
  c_stops_id = 3
  c_afficates_id = 4
  c_fricatives_id = 5
  c_nasals_id = 6
  c_liquids_id = 7
  c_semivowels_id = 8
  c_others = 9

def coarse_phone_to_fine_phone(phone):
  if phone in Phonemes.monophthongs:
    return Phonemes.monophthongs_id
  elif phone in Phonemes.diphthongs:
    return Phonemes.diphthongs_id
  elif phone in Phonemes.r_vowels:
    return Phonemes.r_vowels_id
  elif phone in Phonemes.c_stops:
    return Phonemes.c_stops_id
  elif phone in Phonemes.c_afficates:
    return Phonemes.c_afficates_id
  elif phone in Phonemes.c_fricatives:
    return Phonemes.c_fricatives_id
  elif phone in Phonemes.c_nasals:
    return Phonemes.c_nasals_id
  elif phone in Phonemes.c_liquids:
    return Phonemes.c_liquids_id
  elif phone in Phonemes.c_semivowels:
    return Phonemes.c_semivowels_id
  else:
    return Phonemes.c_others



def get_phoneme_transcript(grapheme_transcript, markers=True):    
  phoneme_transcript = get_phoneme_seq(grapheme_transcript)
  phoneme_transcript = [item if len(item)!=3 else item[:-1] for item in phoneme_transcript]   
  phoneme_transcript = [item for item in phoneme_transcript if item not in {"'"}]
  if markers:
    return [SOS] + phoneme_transcript + [EOS]
  else:
    return phoneme_transcript

def mask_phoneme_tokens(phoneme_sequence, masking_fraction):
  #print(masking_fraction)
  phoneme_sequence = phoneme_sequence[1:-1] # removing start and end, as they do not get masked
  seq_length = len(phoneme_sequence)
  positions = list(range(seq_length))
  random.shuffle(positions)
  mask_positions = positions[0:floor(masking_fraction*seq_length)]
  masked_sequence = [MASK if i in mask_positions else item for i,item in enumerate(phoneme_sequence)]
  masked_sequence = [SOS] + masked_sequence + [EOS]
  masked_positions = [1 if item==MASK else 0 for item in masked_sequence]
  assert len(phoneme_sequence) + 2 == len(masked_sequence) == len(masked_positions)
  return masked_sequence, masked_positions


def convert_phonemes_to_ids(phoneme_sequence):
  return [phone_list.index(item) for item in phoneme_sequence]

def load_phoneme_sequences(json_manifest_paths, remove_duplicates=True):
  sentences = []
  for json_path in json_manifest_paths:
    with open(json_path) as f:
      lines = [line for line in f]
      data = Parallel(n_jobs=-1)(delayed(normalized_json_transcript)(line) for line in tqdm(lines, total=len(lines)))
      print(type(data))
      print(data[0])
      sentences += data
  
  if remove_duplicates:
    sentences = list(set(sentences)) #remove duplicates
  
  phoneme_sentences = Parallel(n_jobs=8)(delayed(get_phoneme_transcript)(item) for item in tqdm(sentences,total=len(sentences)))
  print(type(phoneme_sentences))
  print(phoneme_sentences[0])
  return phoneme_sentences


def collate_fn(batch, masking_fraction, padding_value=0):
  #print(masking_fraction)
  masked_sequences = []
  sequence_lengths = []
  phoneme_sequences = []
  padding_positions = []
  masked_positions = []

  for i,phoneme_sequence in enumerate(batch):
    masked_sequence,masked_position = mask_phoneme_tokens(phoneme_sequence, masking_fraction)
    masked_sequences.append(convert_phonemes_to_ids(masked_sequence))
    phoneme_sequences.append(convert_phonemes_to_ids(phoneme_sequence))
    sequence_lengths.append(len(phoneme_sequence))
    masked_positions.append(masked_position)
    assert len(masked_sequence) == len(phoneme_sequence) == len(masked_position)

  max_length = max(sequence_lengths)
  padding_positions = [[1]*len(item)+[0]*(max_length-len(item)) for item in phoneme_sequences]
  masked_sequences = [item + [padding_value]*(max_length-len(item)) for item in masked_sequences]
  phoneme_sequences = [item + [padding_value]*(max_length-len(item)) for item in phoneme_sequences]
  masked_positions = [item + [padding_value]*(max_length-len(item)) for item in masked_positions]

  return torch.tensor(phoneme_sequences,dtype=torch.int64), \
        torch.tensor(masked_sequences,dtype=torch.int64), \
        torch.tensor(padding_positions, dtype=torch.float), \
        torch.tensor(masked_positions,dtype=torch.float), \
        torch.tensor(sequence_lengths, dtype=torch.int64)

def inference_collate_fn(batch, padding_value=0):
  sequence_lengths = []
  phoneme_sequences = []
  padding_positions = []

  for i,phoneme_sequence in enumerate(batch):
    phoneme_sequences.append(convert_phonemes_to_ids(phoneme_sequence))
    sequence_lengths.append(len(phoneme_sequence))

  max_length = max(sequence_lengths)
  padding_positions = [[1]*len(item)+[0]*(max_length-len(item)) for item in phoneme_sequences]
  phoneme_sequences = [item + [padding_value]*(max_length-len(item)) for item in phoneme_sequences]

  return torch.tensor(phoneme_sequences,dtype=torch.int64), \
        torch.tensor(padding_positions, dtype=torch.float), \
        torch.tensor(sequence_lengths, dtype=torch.int64)

def error_classifier_collate_fn(batch, padding_value=0):
  sequence_lengths = []
  phoneme_sequences = []
  error_sequences = []
  padding_positions = []
  tts_seqs = []
  vowels = []
  fines = []

  for error_sequence, phoneme_sequence, tts_seq, vowel, fine in batch:
    try:
      assert len(phoneme_sequence) == len(tts_seq)
    except:
      print(len(phoneme_sequence))
      print(len(tts_seq))
      raise
    error_sequences.append(error_sequence)
    phoneme_sequences.append(convert_phonemes_to_ids(phoneme_sequence))
    sequence_lengths.append(len(phoneme_sequence))
    tts_seqs.append(tts_seq)
    vowels.append(vowel)
    fines.append(fine)
  
  max_length = max(sequence_lengths)
  padding_positions = [[1]*len(item)+[0]*(max_length-len(item)) for item in phoneme_sequences]
  phoneme_sequences = [item + [padding_value]*(max_length-len(item)) for item in phoneme_sequences]
  error_sequences = [item + [padding_value]*(max_length-len(item)) for item in error_sequences]
  tts_seqs = [item + [padding_value]*(max_length-len(item)) for item in tts_seqs]
  vowels = [item + [padding_value]*(max_length-len(item)) for item in vowels]
  fines = [item + [padding_value]*(max_length-len(item)) for item in fines]

  torch.tensor(tts_seqs, dtype=torch.int64)

  return torch.tensor(phoneme_sequences,dtype=torch.int64), \
        torch.tensor(error_sequences,dtype=torch.int64), \
        torch.tensor(padding_positions, dtype=torch.float), \
        torch.tensor(sequence_lengths, dtype=torch.int64), \
        torch.tensor(tts_seqs, dtype=torch.int64), \
        torch.tensor(vowels, dtype=torch.int64), \
        torch.tensor(fines, dtype=torch.int64)

def get_WER_from_para(para):
  return float(para.strip().split('\n')[2][5:])

def geneate_error_data_from_hypotheses_file(path_hypotheses, is_train=False, skip_zero_CER=False):
  with open(path_hypotheses) as f, open(path_hypotheses.split(".")[0]+"_tts.txt") as r:
    paragraphs = f.read().strip().split('\n\n')
    tts_paragraphs = r.read().strip().split('\n\n')
    ref_hyp_pairs = [(para.strip().split('\n')[3][5:], para.strip().split('\n')[4][5:], tts.strip().split('\n')[4][5:]) \
                      for para, tts in zip(paragraphs, tts_paragraphs) if (not skip_zero_CER or get_WER_from_para(para)!=0.0)]
    pool = Pool(16)
    multiprocessed_output = list(filter(None, pool.map(__generate_error_sequence, tqdm(ref_hyp_pairs, is_train, total=len(ref_hyp_pairs)))))
    pool.close()
    # error_sequences, reference_phonemes = zip(*multiprocessed_output)
    error_sequences, reference_phonemes, tts_sequences, vowels, fines = zip(*multiprocessed_output)
    # print(error_sequences[0])
    # print(reference_phonemes[0])
    # return list(zip(error_sequences, reference_phonemes))
    print(error_sequences[0])
    print(reference_phonemes[0])
    print(tts_sequences[0])
    print(vowels[0])
    print(fines[0])
    return list(zip(error_sequences, reference_phonemes, tts_sequences, vowels, fines))

def __get_error_sequence_between_words(reference, hypothesis):
  p_h = get_phoneme_transcript(hypothesis,markers=False)
  p_r = get_phoneme_transcript(reference,markers=False)
  error_sequence = []
  insert_left = None
  insert_right = None

  diff = ndiff(p_r, p_h)
  for i,e in enumerate(diff):
    if e[0] == ' ':
      error_sequence.append(0)
    elif e[0] == '-':
      error_sequence.append(1)
    elif e[0] == '+':
      if i>0 and diff[i-1][0]=='-':
        assert error_sequence[-1] == 1
        continue
      elif i>0 and i<len(diff)-1 and diff[i-1][0]==' ':
        error_sequence[-1] = 1
      else:
        if i == 0:
          insert_left = 1
        else:
          if not (i == len(diff)-1 and diff[i-1][0]==' '):
            print(list(enumerate(diff)))
            print(i)
          insert_right = 1
    else:
      print('ERROR')
      exit()


  assert len(error_sequence) == len(p_r)
  return error_sequence, insert_left, insert_right


def is_align(reference, hypothesis):
  reference_phonemes = get_phoneme_transcript(reference,markers=False)
  reference = reference.lower()
  hypothesis = hypothesis.lower()
  error_sequence = []

  aligner = PowerAligner(reference, hypothesis, lowercase=True, verbose=False, lexicon="lex/cmudict.rep.json")
  try:
      aligner.align()
  except:
      print("alignement failed")
      return None

  power_alignment = aligner.power_alignment
  # print(power_alignment)
  p_ref = [SOS] + power_alignment.ref() + [EOS]
  p_hyp = [SOS] + power_alignment.hyp() + [EOS]
  p_ops = ['C'] + power_alignment.align + ['C']


  hyp_pointer = 0
  ref_pointer = 0

  ref_invariant = [SOS,' '] + reference_phonemes + [' ',EOS]
  ref_constructed = []

  for op in p_ops:
      if hyp_pointer < len(p_hyp):
          hyp_word = p_hyp[hyp_pointer]
      else:
          assert op=='D'

      if ref_pointer < len(p_ref):
          ref_word = p_ref[ref_pointer]
      else:
          assert op == 'I'

      if not (ref_word in [SOS,EOS]):
        ph_ref_word = get_phoneme_transcript(ref_word, markers=False)
      else:
        ph_ref_word = [ref_word]

      if op == 'C':
          # if ref_word != hyp_word :
          #   print(ref_word)
          #   print(hyp_word)
          assert ref_word == hyp_word
          #tp,tr,th,cm = _get_statistics(ref_word, hyp_word)
          ref_constructed += ph_ref_word
          error_sequence += [0]*len(ph_ref_word)
          if len(error_sequence) < len(reference_phonemes) + 4:
            error_sequence += [0]
            ref_constructed += [' ']

          hyp_pointer +=1
          ref_pointer +=1
      elif op == 'D':
          #tp,tr,th,cm = _get_statistics(ref_word,'')
          error_sequence += [1]*len(ph_ref_word)
          ref_constructed += ph_ref_word
          if len(error_sequence) < len(reference_phonemes) + 4:
            error_sequence += [0]
            ref_constructed += [' ']
          ref_pointer +=1
      elif op == 'I':
          error_sequence[-1] = 1
          #tp,tr,th,cm = _get_statistics('',hyp_word)
          hyp_pointer +=1
      elif op == 'S':
          #tp,tr,th,cm = _get_statistics(ref_word,hyp_word)
          ph_hyp_word = get_phoneme_transcript(hyp_word, markers=False)
          ref_constructed += ph_ref_word
          substition_error, insert_left, insert_right = __get_error_sequence_between_words(ref_word, hyp_word)
          if insert_left:
            error_sequence[-1] = 1
          error_sequence += substition_error
          ref_pointer +=1
          hyp_pointer +=1
          if len(error_sequence) < len(reference_phonemes) + 4:
            error_sequence += [0]
            ref_constructed += [' ']
          if insert_right:
            error_sequence[-1] = 1

  assert error_sequence[0] == error_sequence[-1] == 0
  error_sequence = error_sequence[1:-1]

  reference_phonemes = [SOS] + reference_phonemes + [EOS]
  return len(error_sequence) == len(reference_phonemes)

def __generate_error_sequence(reference_hypothesis_pair, is_train=False):
    reference, hypothesis, tts = reference_hypothesis_pair
    error_sequence = []
    tts_sequence = []
    # print(reference_hypothesis_pair)
    reference_phonemes = get_phoneme_transcript(reference,markers=False)
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    tts = tts.lower()
    if is_train:
      aligner = PowerAligner(reference, hypothesis, lowercase=True, verbose=False, lexicon="lex/cmudict.rep.json")
      try:
          aligner.align()
      except:
          print("alignement failed")
          return None

      power_alignment = aligner.power_alignment
      # print(power_alignment)
      p_ref = [SOS] + power_alignment.ref() + [EOS]
      p_hyp = [SOS] + power_alignment.hyp() + [EOS]
      p_ops = ['C'] + power_alignment.align + ['C']


      hyp_pointer = 0
      ref_pointer = 0

      ref_invariant = [SOS,' '] + reference_phonemes + [' ',EOS]
      ref_constructed = []

      for op in p_ops:
          if hyp_pointer < len(p_hyp):
              hyp_word = p_hyp[hyp_pointer]
          else:
              assert op=='D'

          if ref_pointer < len(p_ref):
              ref_word = p_ref[ref_pointer]
          else:
              assert op == 'I'

          if not (ref_word in [SOS,EOS]):
            ph_ref_word = get_phoneme_transcript(ref_word, markers=False)
          else:
            ph_ref_word = [ref_word]

          if op == 'C':
              # if ref_word != hyp_word :
              #   print(ref_word)
              #   print(hyp_word)
              assert ref_word == hyp_word
              #tp,tr,th,cm = _get_statistics(ref_word, hyp_word)
              ref_constructed += ph_ref_word
              error_sequence += [0]*len(ph_ref_word)
              if len(error_sequence) < len(reference_phonemes) + 4:
                error_sequence += [0]
                ref_constructed += [' ']

              hyp_pointer +=1
              ref_pointer +=1
          elif op == 'D':
              #tp,tr,th,cm = _get_statistics(ref_word,'')
              error_sequence += [1]*len(ph_ref_word)
              ref_constructed += ph_ref_word
              if len(error_sequence) < len(reference_phonemes) + 4:
                error_sequence += [0]
                ref_constructed += [' ']
              ref_pointer +=1
          elif op == 'I':
              error_sequence[-1] = 1
              #tp,tr,th,cm = _get_statistics('',hyp_word)
              hyp_pointer +=1
          elif op == 'S':
              #tp,tr,th,cm = _get_statistics(ref_word,hyp_word)
              ph_hyp_word = get_phoneme_transcript(hyp_word, markers=False)
              ref_constructed += ph_ref_word
              substition_error, insert_left, insert_right = __get_error_sequence_between_words(ref_word, hyp_word)
              if insert_left:
                error_sequence[-1] = 1
              error_sequence += substition_error
              ref_pointer +=1
              hyp_pointer +=1
              if len(error_sequence) < len(reference_phonemes) + 4:
                error_sequence += [0]
                ref_constructed += [' ']
              if insert_right:
                error_sequence[-1] = 1
      
      assert error_sequence[0] == error_sequence[-1] == 0
      error_sequence = error_sequence[1:-1]

      reference_phonemes = [SOS] + reference_phonemes + [EOS]
      if not (len(error_sequence) == len(reference_phonemes)):
        print('len(error_sequence) == len(reference_phonemes)... returning None')
        return None
      assert len(error_sequence) == len(reference_phonemes)
    else:
      reference_phonemes = [SOS] + reference_phonemes + [EOS]

    reference_phonemes_tts = get_phoneme_transcript(reference,markers=False)
    aligner = PowerAligner(reference, tts, lowercase=True, verbose=False, lexicon="lex/cmudict.rep.json")
    try:
        aligner.align()
    except:
        print("alignement failed")
        return None

    power_alignment = aligner.power_alignment
    # print(power_alignment)
    p_ref = [SOS] + power_alignment.ref() + [EOS]
    p_hyp = [SOS] + power_alignment.hyp() + [EOS]
    p_ops = ['C'] + power_alignment.align + ['C']


    hyp_pointer = 0
    ref_pointer = 0

    ref_invariant = [SOS,' '] + reference_phonemes_tts + [' ',EOS]
    ref_constructed = []

    for op in p_ops:
        if hyp_pointer < len(p_hyp):
            hyp_word = p_hyp[hyp_pointer]
        else:
            assert op=='D'

        if ref_pointer < len(p_ref):
            ref_word = p_ref[ref_pointer]
        else:
            assert op == 'I'

        if not (ref_word in [SOS,EOS]):
          ph_ref_word = get_phoneme_transcript(ref_word, markers=False)
        else:
          ph_ref_word = [ref_word]

        if op == 'C':
            # if ref_word != hyp_word :
            #   print(ref_word)
            #   print(hyp_word)
            assert ref_word == hyp_word
            #tp,tr,th,cm = _get_statistics(ref_word, hyp_word)
            ref_constructed += ph_ref_word
            tts_sequence += [0]*len(ph_ref_word)
            if len(tts_sequence) < len(reference_phonemes_tts) + 4:
              tts_sequence += [0]
              ref_constructed += [' ']

            hyp_pointer +=1
            ref_pointer +=1
        elif op == 'D':
            #tp,tr,th,cm = _get_statistics(ref_word,'')
            tts_sequence += [1]*len(ph_ref_word)
            ref_constructed += ph_ref_word
            if len(tts_sequence) < len(reference_phonemes_tts) + 4:
              tts_sequence += [0]
              ref_constructed += [' ']
            ref_pointer +=1
        elif op == 'I':
            tts_sequence[-1] = 1
            #tp,tr,th,cm = _get_statistics('',hyp_word)
            hyp_pointer +=1
        elif op == 'S':
            #tp,tr,th,cm = _get_statistics(ref_word,hyp_word)
            ph_hyp_word = get_phoneme_transcript(hyp_word, markers=False)
            ref_constructed += ph_ref_word
            substition_error, insert_left, insert_right = __get_error_sequence_between_words(ref_word, hyp_word)
            if insert_left:
              tts_sequence[-1] = 1
            tts_sequence += substition_error
            ref_pointer +=1
            hyp_pointer +=1
            if len(tts_sequence) < len(reference_phonemes_tts) + 4:
              tts_sequence += [0]
              ref_constructed += [' ']
            if insert_right:
              tts_sequence[-1] = 1
    
    assert tts_sequence[0] == tts_sequence[-1] == 0
    tts_sequence = tts_sequence[1:-1]
    reference_phonemes_tts = [SOS] + reference_phonemes_tts + [EOS]

    
    if not (len(tts_sequence) == len(reference_phonemes_tts)):
      print('len(tts_sequence) == len(reference_phonemes_tts)... returning None')
      return None
    
    vowel = [1 if item in Phonemes.vowels else 0 for item in reference_phonemes]
    fine = [coarse_phone_to_fine_phone(phone) for phone in reference_phonemes]
    return error_sequence, reference_phonemes, tts_sequence, vowel, fine


if __name__ == '__main__':
  
  s1 = 'anatomy hi hello democracy'
  s2 = 'and that to me hello de mo gracy'

  p1 = get_phoneme_transcript(s1)
  p2 = get_phoneme_transcript(s2)

  error_sequence = __generate_error_sequence((s1,s2))
  

