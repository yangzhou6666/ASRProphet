'''
This file is for greedily selecting sentences with the highest error metrics.
'''
import argparse


def main(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='error model sampling')
    parser.add_argument("--selection_json_file", type=str, help='path to json file from where sentences are selected')
    parser.add_argument("--seed_json_file", type=str, help='path to json file containing seed sentences')
    parser.add_argument("--error_model_weights", type=str, help='weights provided by error model inference')
    parser.add_argument("--random_json_path",type=str,
      help='path to dir containing json files for randomly selected sentences, used to ensure same amount of speech time')
    parser.add_argument("--output_json_path", type=str, 
      help='path to dir containing json files for sentences selected via error model weights')
    parser.add_argument("--exp_id", type=str, help='experiment id')
    args=parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)