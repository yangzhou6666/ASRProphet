import os
import sys
import json
sys.path.append("..")
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
import argparse
import random
import string
from tqdm import tqdm
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train Word Error Predictor')
    parser.add_argument("--sampling_method", default='word_enhance', type=str,
                        required=True, help='sampling method')
    parser.add_argument("--seed_json_file", type=str,
                        required=True, help='path to seed json file')
    parser.add_argument("--selection_json_file", type=str,
                        required=True, help='path to input file')
    parser.add_argument("--data_folder", type=str,
                        required=True, help='path to input file')
    parser.add_argument("--finetuned_ckpt", default=None,
                        type=str, help='path to finetuned ckpt')
    parser.add_argument("--seed", default=1, type=int, help='seed id')
    parser.add_argument("--output_json_path", type=str,
                        required=True, help='json fpath to save the ranked texts')
    args=parser.parse_args()
    return args

def get_f(freq,tau=500):
    return 1 - np.exp(-freq/tau)

def format_data(json_fpath):
    file = open(json_fpath)
    data = file.readlines()
    file.close()

    texts = []
    fpaths = []
    durations = []
    for d in data:
        js = json.loads(d)
        texts.append(js["text"])
        fpaths.append(js["audio_filepath"])
        durations.append(js["duration"])

    return texts, fpaths, durations


def sort(texts, fpaths, probabilities, durations):
    list1 = probabilities
    indexs = [i for i in range(len(probabilities))]
    list1, indexs = zip(*sorted(zip(list1, indexs)))

    indexs = list(indexs)[::-1]  # reverse

    texts = list(map(texts.__getitem__, indexs))
    fpaths = list(map(fpaths.__getitem__, indexs))
    probabilities = list(map(probabilities.__getitem__, indexs))
    durations = list(map(durations.__getitem__, indexs))

    return texts, fpaths, probabilities, durations

def dump_samples(samples, filename):
    output_dir = os.path.split(filename)[0]
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(sample)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class WordErrorSampler():
    '''
    The class for selecting test cases using the word-level error predictor
    '''
    def __init__(self, finetuned_ckpt, selection_json_file) -> None:
        '''load required parameters'''

        # path to the finetuned model
        self.finetuned_ckpt = finetuned_ckpt
        # path to the selection json file (the json file that contains the test cases)
        self.selection_json_file = selection_json_file



    def load_model(self):
        '''Load tokenizer and model'''

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_ckpt)
        # load the model
        self.model = AutoModelForTokenClassification.from_pretrained(self.finetuned_ckpt)
        self.model.cuda()
        self.model.eval()

    def load_sampling_data(self):
        '''Load the data for sampling'''

        # load the selection data
        self.selection_texts, self.selection_fpaths, self.selection_durations = format_data(self.selection_json_file)

        # tokenize the selection data
        test_encodings = self.tokenizer(self.selection_texts, truncation=True)
        test_dataset = TextDataset(test_encodings)

        # prepare data loader
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False)

        self.res = []
        # score for each test case
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            outputs = self.model(
                input_ids, attention_mask)

            probs = F.softmax(outputs.logits, -1)
            error_probs = probs[:, :, 1]
            self.res.append(error_probs[0].cpu().detach().numpy().tolist())
    
    def normal_sample(self, required_duration: float):
        '''select the test cases based on sum of their error score'''
        score = [sum(n)/len(n) for n in self.res]

        # rank the test cases based on the score
        test_texts, fpaths, score, durations = sort(self.selection_texts, self.selection_fpaths, score, self.selection_durations)
        samples = []
        selected_duration = 0.0

        for text, fpath, duration in zip(test_texts, fpaths, durations):
            samples.append(json.dumps({"text": text, "audio_filepath": fpath, "duration": duration}) + "\n")
            selected_duration += json.loads(samples[-1].strip())['duration']
            if selected_duration >= required_duration:
                break

        print('sampled {} samples...'.format(len(samples)))

        return samples

    def sample_with_sub_word_enhancing(self, required_duration: float):
        '''select the test cases based on sum of their error score and consider the subword diversisty enhancing'''

        # we cannot pre-compute a static metric for all texts in advance
        # we need to compute it on the fly

        self.get_sentence_wise_subword_freqs()
        # initialize the all the frequency inforamtion for each run.

        # rank the test cases based on the score
        error_score = self.res
        test_texts, fpaths, error_score, durations = sort(self.selection_texts, self.selection_fpaths, error_score, self.selection_durations)

        samples = [] # store the selected test cases
        selected_duration = 0.0 # accumulative duration of selected test cases

        path_score = {} # store the error score for each text (identified by path)
        for fpath, this_score in zip(fpaths, error_score):
            path_score[fpath] = this_score

        while (selected_duration < required_duration and len(test_texts) > 0):
            # ternimation condition: select enough duration + there are no more test cases

            # first, start with the case with the highest score
            text = test_texts.pop(0)
            fpath = fpaths.pop(0)
            this_score = error_score.pop(0)
            duration = durations.pop(0)

            samples.append(json.dumps({"text": text, "audio_filepath": fpath, "duration": duration}) + "\n")
            selected_duration += duration # update the total duration

            # after selecting a test case, we need to update the sub word frequency
            self.acc_subtoken_freqs += self.sent_wise_subtoken_freqs[fpath]['freq']

            # update the score for the remaining test cases
            new_scores = []
            for i in range(len(test_texts)):
                new_path = fpaths[i]

                text_subwords = self.tokenizer.encode(test_texts[i])
                subwords_weights = np.zeros(len(self.subtoken_vocab))
                for index, n in enumerate(text_subwords):
                    subwords_weights[n] = path_score[new_path][index]
                f_old = get_f(self.acc_subtoken_freqs)
                f_new = get_f(self.acc_subtoken_freqs + self.sent_wise_subtoken_freqs[new_path]['freq']) # the new frequency if this test case is selected
                new_score = (f_new - f_old) * subwords_weights
                # the score that consider the sub-word diversity enhancing
                new_score = np.sum(new_score)
                new_scores.append(new_score)
            
            assert len(new_scores) == len(test_texts)
            test_texts, fpaths, error_score, durations = sort(test_texts, fpaths, new_scores, durations)

        return samples




    def get_subtokens(self, text):
        '''compute the sub token frequency of a text'''
        subtoken_list = self.tokenizer.tokenize(text)
        return subtoken_list

    def get_sentence_wise_subword_freqs(self):
        '''compute the sub word frequency of each sentence'''
        self.subtoken_vocab = self.tokenizer.vocab
        self.acc_subtoken_freqs = np.zeros(len(self.subtoken_vocab)) 
        # vector to store the accumulated sub word frequency
        self.sent_wise_subtoken_freqs = {} 
        # a dictionary to store the sub word frequency of each sentence
        for text, path in zip(self.selection_texts, self.selection_fpaths):
            # for a text (and its path)
            freq = np.zeros(len(self.subtoken_vocab))
            subtokens = self.get_subtokens(text) # get the subtoken sequence
            for subtoken in subtokens:
                freq[self.subtoken_vocab[subtoken]] +=1
                # update the frequency of each sub token
            self.sent_wise_subtoken_freqs[path] = {'freq': freq, 'subtokens': subtokens, 'text': text} # store the subtoken frequency of each sentence

        return self.sent_wise_subtoken_freqs


def compute_required_duration(seed_json_file, random_json_file):
    '''compute the required duration for samping new test cases'''
    random_samples_duration = sum([json.loads(line.strip())['duration'] for line in open(random_json_file)])
    seed_duration = sum([json.loads(line.strip())['duration'] for line in open(seed_json_file)])
    required_duration = random_samples_duration - seed_duration

    return required_duration


def main(args):
    '''
    Function to train and save the model.
    '''
    # set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    err_sampler = WordErrorSampler(args.finetuned_ckpt, args.selection_json_file)
    # load the model
    err_sampler.load_model()
    # load the data
    err_sampler.load_sampling_data()
    # sample the test cases

    sampling_method = args.sampling_method

    for num_sample in [100, 200, 300, 400]:
        random_json_file = os.path.join(args.data_folder, str(num_sample), "seed_" + str(args.seed), "train.json")
        required_duration = compute_required_duration(args.seed_json_file, random_json_file)
        assert required_duration > 0

        if sampling_method == "word_enhance":
            samples = err_sampler.sample_with_sub_word_enhancing(required_duration)
        elif sampling_method == "no_word_enhance":
            samples = err_sampler.normal_sample(required_duration)
        else:
            raise NotImplementedError

        # dump the samples to a file
        output_json_file = os.path.join(args.output_json_path, str(
            num_sample), sampling_method, 'seed_' + str(args.seed), 'train.json')
        seed_samples = [line for line in open(args.seed_json_file)]
        # merge the seed samples with the new samples

        dump_samples(seed_samples + samples, output_json_file)
        print(f"{len(seed_samples + samples)} samples are dumped to {output_json_file}")

        output_json_file = os.path.join(args.output_json_path, str(
            num_sample), sampling_method, 'seed_' + str(args.seed), 'train_no_seed.json')
        seed_samples = [line for line in open(args.seed_json_file)]
        # merge the seed samples with the new samples

        dump_samples(samples, output_json_file)
        print(f"{len(samples)} samples are dumped to {output_json_file}")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_ckpt)
    main(args)

    

