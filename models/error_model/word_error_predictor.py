import sys
sys.path.append("..")

from quartznet_asr.metrics import __levenshtein, word_error_rate
from power import Levenshtein, ExpandedAlignment
from power.aligner import PowerAligner
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import re
from datasets import Dataset
import pandas as pd
import numpy as np
import argparse
import random

def get_label(path: str):
    '''
    Given the path to the reference and transcriptions
    return the model inputs and correpsonding labels
    '''
    inputs = []
    labels = []
    with open(path, 'r') as f:
        for chunk in f.read().strip().split('\n\n'):
            data = chunk.split('\n')
            WER = float(data[1][5:])
            CER = float(data[2][5:])
            ref = data[3][5:]
            hyp = data[4][5:]

            # convert string to list of words
            re_expr = "\'| "
            ref = [x for x in re.split(re_expr, ref.strip()) if x]
            hyp = [x for x in re.split(re_expr, hyp.strip()) if x]

            lev = Levenshtein.align(ref, hyp) 
            lev.editops()
            alignment = lev.expandAlign()

            '''
            The data structure of alignment:
            REF:  anatomy         hi  hello  democracy  
            HYP:  and that to me      hello  de mo gracy
            Eval: S               D   C      S    
            '''

            '''
            Explaination for symbols:
            S: Single word subsitutions
            D: Deletion
            I: Insertion, a hyoothesis word with no aligned reference word
            C: Correct.
            '''

            label = []
            for x in alignment.align:
                if x == 'I':
                    continue
                elif x == 'C':
                    label.append(0)
                else:
                    label.append(1)
            assert len(label) == len(alignment.ref())
            labels.append(label)
            inputs.append(alignment.ref())
    
    return inputs, labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)
    label_all_tokens = True

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_dataset(path: str):
    inputs, labels = get_label(path)
    df = pd.DataFrame({"text": inputs, "labels": labels})
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    return tokenized_datasets

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # compute the accuracy
    cor_cnt = 0
    tot_cnt = 0
    for i in range(len(true_predictions)):
        for j in range(len(true_predictions[i])):
            if true_predictions[i][j] == true_labels[i][j]:
                cor_cnt += 1
            tot_cnt += 1
    acc = cor_cnt / tot_cnt

    # compute precision
    cor_1_cnt = 0
    tot_1_cnt = 0.01
    for i in range(len(true_predictions)):
        for j in range(len(true_predictions[i])):
            if true_predictions[i][j] == 1:
                tot_1_cnt += 1
                if true_labels[i][j] == 1:
                    cor_1_cnt += 1
    precision = cor_1_cnt / tot_1_cnt


    return {"accuracy": acc, "precision": precision}

def parse_args():
    parser = argparse.ArgumentParser(description='Train Word Error Predictor')
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--train_path", type=str, required=True, help='path to training data')
    parser.add_argument("--test_path", type=str, required=True, help='path to testing data')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--output_dir",default=None,type=str,help='path to store the predictor')
    parser.add_argument("--model_name",default="bert-base-uncased",type=str,help='name of the pre-train model')
    args=parser.parse_args()
    return args


def main(args):
    '''
    Function to train and save the model.
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare the dataset
    train_dataset = prepare_dataset(args.train_path)
    test_dataset = prepare_dataset(args.test_path)
    print("Data loaded.")

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=3)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # train the model
    training_args = TrainingArguments(
        f"word_error_predictor",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # save the model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    main(args)

