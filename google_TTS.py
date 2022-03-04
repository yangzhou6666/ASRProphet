import os
import google
from google.cloud import texttospeech

TTS_CONFIG = {'en-GB-Wavenet-A': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-GB-Wavenet-B': {'language_code': 'en-GB', 'gender': 'MALE'}, 'en-GB-Wavenet-C': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-GB-Wavenet-D': {'language_code': 'en-GB', 'gender': 'MALE'}, 'en-GB-Wavenet-F': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-US-Wavenet-G': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Wavenet-H': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Wavenet-I': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Wavenet-J': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Wavenet-A': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Wavenet-B': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Wavenet-C': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Wavenet-D': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Wavenet-E': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Wavenet-F': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-AU-Wavenet-A': {'language_code': 'en-AU', 'gender': 'FEMALE'}, 'en-AU-Wavenet-B': {'language_code': 'en-AU', 'gender': 'MALE'}, 'en-AU-Wavenet-C': {'language_code': 'en-AU', 'gender': 'FEMALE'}, 'en-AU-Wavenet-D': {'language_code': 'en-AU', 'gender': 'MALE'}, 'en-IN-Wavenet-D': {'language_code': 'en-IN', 'gender': 'FEMALE'}, 'en-IN-Wavenet-A': {'language_code': 'en-IN', 'gender': 'FEMALE'}, 'en-IN-Wavenet-B': {'language_code': 'en-IN', 'gender': 'MALE'}, 'en-IN-Wavenet-C': {'language_code': 'en-IN', 'gender': 'MALE'}, 'en-AU-Standard-A': {'language_code': 'en-AU', 'gender': 'FEMALE'}, 'en-AU-Standard-B': {'language_code': 'en-AU', 'gender': 'MALE'}, 'en-AU-Standard-C': {'language_code': 'en-AU', 'gender': 'FEMALE'}, 'en-AU-Standard-D': {'language_code': 'en-AU', 'gender': 'MALE'}, 'en-GB-Standard-A': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-GB-Standard-B': {'language_code': 'en-GB', 'gender': 'MALE'}, 'en-GB-Standard-C': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-GB-Standard-D': {'language_code': 'en-GB', 'gender': 'MALE'}, 'en-GB-Standard-F': {'language_code': 'en-GB', 'gender': 'FEMALE'}, 'en-IN-Standard-D': {'language_code': 'en-IN', 'gender': 'FEMALE'}, 'en-IN-Standard-A': {'language_code': 'en-IN', 'gender': 'FEMALE'}, 'en-IN-Standard-B': {'language_code': 'en-IN', 'gender': 'MALE'}, 'en-IN-Standard-C': {'language_code': 'en-IN', 'gender': 'MALE'}, 'en-US-Standard-A': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Standard-B': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Standard-C': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Standard-D': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Standard-E': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Standard-F': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Standard-G': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Standard-H': {'language_code': 'en-US', 'gender': 'FEMALE'}, 'en-US-Standard-I': {'language_code': 'en-US', 'gender': 'MALE'}, 'en-US-Standard-J': {'language_code': 'en-US', 'gender': 'MALE'}}

def list_voices():
    """Lists the available voices."""

    client = texttospeech.TextToSpeechClient()

    tts_config = {}

    # Performs the list voices request
    voices = client.list_voices()

    for voice in voices.voices:
        # Display the voice's name. Example: tpc-vocoded
        print(f"Name: {voice.name}")

        # Display the supported language codes for this voice. Example: "en-US"
        for language_code in voice.language_codes:
            print(f"Supported language: {language_code}")

        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender)

        # Display the SSML Voice Gender
        print(f"SSML Voice Gender: {ssml_gender.name}")

        # Display the natural sample rate hertz for this voice. Example: 24000
        print(f"Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}\n")

        if 'en-' in language_code:
            tts_config[voice.name] = {"language_code": language_code, "gender": ssml_gender.name}
    
    return tts_config




def synthesize_text(text, output_path, config_name, language_code, gender):
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=config_name,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if gender == 'female' else texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    
    output_path = output_path.replace(".wav", ".mp3")

    # The response's audio_content is binary.
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_path}"')

if __name__=='__main__':
    config_name = 'en-US-Standard-H'
    language_code = TTS_CONFIG[config_name]["language_code"]
    gender = TTS_CONFIG[config_name]["gender"]

    path_to_dataset = "./data/l2arctic/l2arctic_release_v5"
    selected_sub_dataset = ["ASI", "RRBI"]

    api_call_limit = 1000000
    word_count = 0
    for sub_dataset in os.listdir(path_to_dataset):

        if sub_dataset in selected_sub_dataset:

            if len(sub_dataset.split('.')) > 1:
                continue
            
            # path_to_synthetic = os.path.join(path_to_dataset, sub_dataset, 'synthetic_wav', '-'.join([config_name, gender]))
            path_to_synthetic = os.path.join(path_to_dataset, sub_dataset, 'TTS')
            path_to_text = os.path.join(path_to_dataset, sub_dataset, 'transcript')
            
            os.makedirs(path_to_synthetic, exist_ok=True)

            
            for text_id in os.listdir(path_to_text):
                with open(os.path.join(path_to_text, text_id)) as f:
                    transcript = f.read()
                    word_count += len(transcript.split())
                    if word_count > api_call_limit:
                        print("Reach API call limits.")
                        exit()
                    path_to_save_audio = os.path.join(path_to_synthetic, text_id.split('.')[0] + ".wav")
                    if not os.path.exists(path_to_save_audio):
                        # if query APIs if the file is not generated.
                        synthesize_text(transcript, path_to_save_audio, config_name, language_code, gender)


