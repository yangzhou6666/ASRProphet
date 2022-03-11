import subprocess
import sys
import os
import json
import warnings
import argparse
import pandas as pd


from numpy.core.numeric import cross
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def finetune_deepspeech(train_csv_path, test_csv_path, gpu_id, checkpoint_dir, export_dir, model_scorer, num_epochs, learning_rate):
    
    cmd = f"docker exec -it gpu{gpu_id}-deepspeech sh -c '" + \
            "python -u DeepSpeech.py --noshow_progressbar" + \
            f"    --train_cudnn" + \
            f"    --train_files {train_csv_path}" + \
            f"    --dev_files {test_csv_path}" + \
            f"    --test_files {test_csv_path}" + \
            f"    --train_batch_size 16" + \
            f"    --test_batch_size 16" + \
            f"    --n_hidden 2048" + \
            f"    --epochs {num_epochs}" + \
            f"    --learning_rate {learning_rate}" + \
            f"    --checkpoint_dir {checkpoint_dir}" + \
            f"    --export_dir {export_dir}/" + \
            f"    --scorer {model_scorer};" + \
            f" ./convert_graphdef_memmapped_format --in_graph={export_dir}/output_graph.pb --out_graph={export_dir}/output_graph.pbmm'"

    print(cmd)

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    output = out.decode("utf-8").split("\n")
    for o in output:
        print(o)
    #     f.write(o + "\n")

def create_csv_finetuning_format(json_fpath, csv_fpath, wav_dir):
    
    wav_paths = []
    references = []

    file = open(json_fpath)

    for line in file.readlines():
        js_instance = json.loads(line)

        wav_path = os.path.join(wav_dir, js_instance["audio_filepath"])
        reference = js_instance["text"].lower()

        wav_paths.append(wav_path)
        references.append(reference)

    file.close()

    df = pd.DataFrame(data={"wav_filename": wav_paths, "transcript": references})
    df.to_csv(csv_fpath, index=False)



def parse_args():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument("--batch_size", default=16,
                        type=int, help='data batch size')
    parser.add_argument("--train_manifest", type=str, required=True,
                        help='relative path given dataset folder of training manifest file')
    parser.add_argument("--val_manifest", type=str, required=True,
                        help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--wav_dir", type=str, required=True,
                        help='patht to the directory containing wav files')
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help='deepspeech checkpoint dir to load and save')
    parser.add_argument("--model_scorer", type=str, required=True,
                        help='deepspeech model scorer')
    parser.add_argument("--export_dir", type=str, required=True,
                        help='saves results in this directory')
    parser.add_argument("--gpu_id", required=True, type=int, help='gpu id for experiment')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--num_epochs", default=100, type=int,
                        help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--learning_rate", default=1e-4, type=float, help='learning rate')

    args = parser.parse_args()
    return args


def main(args):

    os.makedirs(args.export_dir, exist_ok=True)

    train_csv_path = os.path.join(args.export_dir, "train.csv")
    test_csv_path = os.path.join(args.export_dir, "test.csv")
    
    create_csv_finetuning_format(args.train_manifest, train_csv_path, args.wav_dir)
    create_csv_finetuning_format(args.val_manifest, test_csv_path, args.wav_dir)

    finetune_deepspeech(train_csv_path, test_csv_path, args.gpu_id, args.checkpoint_dir,
                        args.export_dir, args.model_scorer, args.num_epochs, args.learning_rate)

if __name__ == "__main__":

    args = parse_args()
    # print_dict(vars(args))

    main(args)


    
