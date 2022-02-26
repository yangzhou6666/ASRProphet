import sys
sys.path.append("..")

from quartznet_asr.metrics import __levenshtein, word_error_rate
from power import Levenshtein, ExpandedAlignment
from power.aligner import PowerAligner
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import re
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np


def get_label(path: str):
    '''
    Given the path to the reference and transcriptions
    return the model inputs and correpsonding labels
    '''
    inputs = []
    labels = []
    with open(path_to_result, 'r') as f:
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

    # results = metric.compute(predictions=true_predictions, references=true_labels)
    # compute the accuracy
    cor_cnt = 0
    tot_cnt = 0
    for i in range(len(true_predictions)):
        for j in range(len(true_predictions[i])):
            if true_predictions[i][j] == true_labels[i][j]:
                cor_cnt += 1
            tot_cnt += 1
    acc = cor_cnt / tot_cnt

    # To-do: recall, f1, accuracy
    return {"accuracy": acc}

if __name__ == "__main__":
    path_to_result = '/workspace/data/l2arctic/processed/ASI/manifests/train/quartznet/error_model_tts/50/seed_1/test_out_ori.txt'
    inputs, labels = get_label(path_to_result)

    print(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    df = pd.DataFrame({"text": inputs, "labels": labels})
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    print("Data loaded.")


    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)

    metric = load_metric("seqeval")

    args = TrainingArguments(
        f"word_error_predictor",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

