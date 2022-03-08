import subprocess
import os


def convert_mp3_to_wav(input_fpath, output_fpath):
    setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
    os.system(f"ffmpeg -i {input_fpath} {setting} {output_fpath} -y")


if __name__ == "__main__" :

    work_dir = "./data/l2arctic/l2arctic_release_v5"
    selected_sub_data = ["ASI", "RRBI"]

    for subfolder in os.listdir(work_dir):
        if not len(subfolder.split('.')) == 1:
            continue
        if subfolder in selected_sub_data:
            print(subfolder)
            for file in os.listdir(os.path.join(work_dir, subfolder, 'TTS')):
                original_file = os.path.join(work_dir, subfolder, 'TTS', file)
                output_file = original_file.replace(".mp3", ".wav")
                convert_mp3_to_wav(original_file, output_file)
