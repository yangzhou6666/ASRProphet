import os
import json
from gtts import gTTS
from tqdm import tqdm

def googleGenerateAudio(text, audio_path):
    tempfile = audio_path + "_temp.mp3"
    googleTTS = gTTS(text, lang='en-us')
    googleTTS.save(tempfile)
    setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
    os.system(f"ffmpeg -loglevel error -i {tempfile} {setting} {audio_path} -y")
    os.remove(tempfile)

def rvGenerateAudio(text, audio_path):
    tempfile = audio_path + "_temp.mp3"
    cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + tempfile
    os.system(cmd)
    setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
    os.system(f"ffmpeg -loglevel error -i {tempfile} {setting} {audio_path} -y")
    os.remove(tempfile)
    
def macGenerateAudio(text, audio_path):
    tempfile = audio_path + "_temp.mp3"
    cmd = "say -v Alex -o {tempfile} \"{text}\""
    os.system(cmd)
    setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
    os.system(f"ffmpeg -loglevel error -i {tempfile} {setting} {audio_path} -y")
    os.remove(tempfile)

os.makedirs("/workspace/ASRDebugger/data/LibriSpeech/TTS/")

with open("./manifests/seed_plus_dev.json") as f, open("./TTS/seed_plus_dev.json", "a") as w:
    count = 0
    texts = []
    for line in f.readlines():
        item = json.loads(line)
        texts.append(item["text"])

    for index, text in enumerate(tqdm(texts)):
        googleGenerateAudio(text, "/workspace/ASRDebugger/data/LibriSpeech/TTS/"+str(index)+".wav")
        json.dump({"id": index, "text": text, "audio_filepath": "/workspace/ASRDebugger/data/LibriSpeech/TTS/"+str(index)+".wav"}, w)
        w.write("\n")


