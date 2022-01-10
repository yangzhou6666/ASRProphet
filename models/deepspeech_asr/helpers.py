import string
import re
import jiwer
from enum import Enum
from metrics import word_error_rate, f_wer, f_cer
import json
from normalise import normalise, tokenize_basic


def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)


def print_sentence_wise_wer(hypotheses, references, output_file, input_file):
    
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
        for hyp, ref, wer, cer, wav_filename in zip(hypotheses, references, wers, cers, wav_filenames):
            f.write(wav_filename+"\n")
            f.write("WER: "+str(wer)+'\n')
            f.write("CER: " + str(cer) + '\n')
            f.write("Ref: "+ref+'\n')
            f.write("Hyp: "+hyp+'\n')
            f.write('\n')

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
