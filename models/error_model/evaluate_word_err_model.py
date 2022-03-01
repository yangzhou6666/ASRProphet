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
from word_error_predictor import compute_metrics, prepare_dataset



def parse_args():
    parser = argparse.ArgumentParser(description='Train Word Error Predictor')
    parser.add_argument("--train_path", type=str, required=True, help='path to testing data')
    parser.add_argument("--test_path", type=str, required=True, help='path to testing data')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--finetuned_ckpt", default=None,
                        type=str, help='path to finetuned ckpt')
    args=parser.parse_args()
    return args


def evalaute(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare the dataset
    test_dataset = prepare_dataset(args.test_path)
    print("Data loaded.")

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_ckpt)
    model = AutoModelForTokenClassification.from_pretrained(args.finetuned_ckpt, num_labels=3)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # train the model
    training_args = TrainingArguments(
        f"word_error_predictor",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    result = trainer.evaluate()
    print(result)

if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_ckpt)
    evalaute(args)

