import wave
import contextlib

def measure_audio_duration(filepath:str)->int:
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

if __name__ == "__main__" :

    filepath = '/media/mhilmiasyrofi/ASRDebugger/data/LibriSpeech/test-clean/61/70968/61-70968-0000.wav'
    duration = measure_audio_duration(filepath)

    print(duration)

    
