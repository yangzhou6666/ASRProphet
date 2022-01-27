# Source:  https://huggingface.co/blog/fine-tune-wav2vec2-english

from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2ForCTC
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
import json
import re
from datasets import load_dataset, load_metric
import numpy as np
import argparse


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]}
                        for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                        for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):

    audio_array, _ = sf.read(batch["audio_filepath"])

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio_array, sampling_rate=16000).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -
                    100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning Wav2Vec2')
    parser.add_argument("--batch_size", default=32,
                        type=int, help='data batch size')
    parser.add_argument("--train_manifest", type=str, required=True,
                        help='relative path given dataset folder of training manifest file')
    parser.add_argument("--val_manifest", type=str, required=True,
                        help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='saves results in this directory')
    parser.add_argument("--vocab", type=str, required=True,
                        help='saves vocab.json for tokenizer')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--num_epochs", default=30, type=int,
                        help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--learning_rate", default=1e-4,
                        type=float, help='learning rate')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    data_files = {"train": args.train_manifest,
                "test": args.val_manifest}
    
    data = load_dataset("json", data_files=data_files)

    data.remove_columns("duration")
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    data = data.map(remove_special_characters)


    vocabs = data.map(extract_all_chars, batched=True, batch_size=-1,
                    keep_in_memory=True, remove_columns=data.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0])
                    | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    with open(args.vocab, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # tokenizer = Wav2Vec2CTCTokenizer(
    #     args.vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    # feature_extractor = Wav2Vec2FeatureExtractor(
    #     feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

    # processor = Wav2Vec2Processor(
    #     feature_extractor=feature_extractor, tokenizer=tokenizer)

    # data_prepared = data.map(
    #     prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)
    
    # data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    # wer_metric = load_metric("wer")

    # model = Wav2Vec2ForCTC.from_pretrained(
    #     "facebook/wav2vec2-base",
    #     ctc_loss_reduction="mean",
    #     pad_token_id=processor.tokenizer.pad_token_id,
    # )
    
    # model.freeze_feature_extractor()

    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     group_by_length=True,
    #     per_device_train_batch_size=args.batch_size,
    #     evaluation_strategy="steps",
    #     num_train_epochs=args.num_epochs,
    #     fp16=True,
    #     gradient_checkpointing=True,
    #     save_steps=500,
    #     eval_steps=500,
    #     logging_steps=500,
    #     learning_rate=args.learning_rate,
    #     weight_decay=0.005,
    #     warmup_steps=1000,
    #     save_total_limit=2,
    # )

    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=data_prepared["train"],
    #     eval_dataset=data_prepared["test"],
    #     tokenizer=processor.feature_extractor,
    # )
    
    # trainer.train()

