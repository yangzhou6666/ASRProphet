import string
import re
import jiwer
from enum import Enum
from typing import List
from metrics import word_error_rate, f_wer, f_cer
import json
from normalise import normalise, tokenize_basic
import Levenshtein as Lev


def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)

def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    wer_list = []
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
        wer_list.append((1.0*__levenshtein(h_list, r_list))/(len(r_list)+1e-20))
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer, wer_list, scores, words


def print_sentence_wise_wer(hypotheses, references, output_file, input_file):
    
    ## ensure that the number of transcriptions are equal to the number of ground-truth texts
    assert len(hypotheses) == len(references)


    wav_filenames = []
    with open(input_file, "r", encoding="utf-8") as f:
        wav_filenames = [json.loads(line.strip())[
            "audio_filepath"] for line in f]

    ## ensure that all audio files are processed
    assert len(hypotheses) == len(wav_filenames)

    WER, wer_list, scores, num_words = word_error_rate(hypotheses=hypotheses, references=references)
    CER, _,_,_ = word_error_rate(hypotheses=hypotheses, references=references, use_cer=True)

    wers = []
    cers = []

    for hyp, ref in zip(hypotheses, references):
        wers.append(f_wer(hyp, ref))
        cers.append(f_cer(hyp, ref))

    with open(output_file, 'w') as f:
        for hyp, ref, wer, cer, wav_filename in zip(hypotheses, references, wers, cers, wav_filenames):
            f.write(wav_filename+"\n")
            f.write("WER: "+str(wer)+'\n')
            f.write("CER: " + str(cer) + '\n')
            f.write("Ref: "+ref+'\n')
            f.write("Hyp: "+hyp+'\n')
            f.write('\n')
    
    print('\n')
    print('\n')
    WER = 100 * WER
    CER = 100 * CER
    print("\n\n==========>>>>>>Evaluation Greedy WER: {:.2f}\n".format(WER))
    print("\n\n==========>>>>>>Evaluation Greedy CER: {:.2f}\n".format(CER))

    return wav_filenames


def print_sentence_wise_wer_with_tts(hypotheses, references, tts_preds, output_file, input_file):

    ## ensure that the number of transcriptions are equal to the number of ground-truth texts
    assert len(hypotheses) == len(references)

    wav_filenames = []
    with open(input_file, "r", encoding="utf-8") as f:
        wav_filenames = [json.loads(line.strip())[
            "audio_filepath"] for line in f]

    ## ensure that all audio files are processed
    assert len(hypotheses) == len(wav_filenames)

    wers = []
    cers = []

    for hyp, ref in zip(hypotheses, references):
        wers.append(f_wer(hyp, ref))
        cers.append(f_cer(hyp, ref))

    with open(output_file, 'w') as f:
        for hyp, ref, wer, cer, wav_filename, tts in zip(hypotheses, references, wers, cers, wav_filenames, tts_preds):
            f.write(wav_filename+"\n")
            f.write("WER: "+str(wer)+'\n')
            f.write("CER: " + str(cer) + '\n')
            f.write("Ref: "+ref+'\n')
            f.write("Hyp: "+hyp+'\n')
            f.write("TTS: "+tts+'\n')
            f.write('\n')

        print('\n')
        print('\n')
        print('================================ \n')
        print("Average WER: " + str(sum(wers)/len(wers)) + '\n')
        print("Average CER: " + str(sum(cers)/len(cers)) + '\n')

    return wav_filenames


def remove_hex(text: str) -> str:
    """
    Example: 
    "\xe3\x80\x90Hello \xe3\x80\x91 World!"
    """
    res = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i+1 < len(text) and text[i+1] == "x":
            i += 3
            res.append(" ")
        else:
            res.append(text[i])
        i += 1
    return "".join(res)


def remove_punctuation(text:str) ->str:
    return text.translate(str.maketrans('', '', string.punctuation))


def normalize_text(text: str) -> str:
    return " ".join(normalise(text, tokenizer=tokenize_basic, verbose=False))


## TODO check missus and mister again
def substitute_word(text: str) -> str:
    """
    word subsitution to make it consistent
    """
    words = text.split(" ")
    preprocessed = []
    for w in words:
        substitution = ""
        if w == "mister":
            substitution = "mr"
        elif w == "missus":
            substitution = "mrs"
        else:
            substitution = w
        preprocessed.append(substitution)
    return " ".join(preprocessed)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = remove_hex(text)
    text = remove_punctuation(text)

    ## it takes long time to normalize
    ## skip this first
    try:
        text = normalize_text(text)
    except:
        text = text

    text = remove_punctuation(text)
    text = substitute_word(text)
    text = jiwer.RemoveMultipleSpaces()(text)
    text = jiwer.ExpandCommonEnglishContractions()(text)
    text = jiwer.RemoveWhiteSpace(replace_by_space=True)(
        text)  # must remove trailing space after it
    text = jiwer.Strip()(text)
    return text

def preprocess_empty_text(text: str) -> str:
    text = remove_hex(text)
    text = remove_punctuation(text)
    text = jiwer.RemoveMultipleSpaces()(text)
    text = jiwer.RemoveWhiteSpace(replace_by_space=True)(
        text)  # must remove trailing space after it
    text = jiwer.Strip()(text)
    return text


def is_empty_file(fpath: str) -> bool:
    file = open(fpath)
    line = file.readline()
    line = line
    file.close()
    if preprocess_empty_text(line) == "":
        return True
    return False
