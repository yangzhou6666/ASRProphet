import json
import wave
import contextlib
import string, re
import jiwer
from normalise import normalise, tokenize_basic

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


def remove_punctuation(text: str) -> str:
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


def measure_audio_duration(filepath: str) -> int:
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def write_json_data(filepath, data):
    with open(filepath, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")

if __name__ == "__main__":

    filepath = '/media/mhilmiasyrofi/ASRDebugger/data/LibriSpeech/test-clean/61/70968/61-70968-0000.wav'
    duration = measure_audio_duration(filepath)

    print(duration)