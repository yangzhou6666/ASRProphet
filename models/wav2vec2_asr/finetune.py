import os
import torch
import random
import helpers
import argparse
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset, Audio
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='wav2vec2')

    parser.add_argument("--wav_dir", type=str, help='directory to the wav file')
    parser.add_argument("--train_manifest", type=str, required=True, help='relative path to train dataset manifest file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path to evaluation dataset manifest file')
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--model", default="wav2vec", type=str)
    parser.add_argument("--seed", default=42, type=int, help='seed')
    return parser.parse_args()


def main(args):
    models = {"wav2vec": "facebook/wav2vec2-base-960h",
              "hubert": "facebook/hubert-large-ls960-ft"}

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(models[args.model])
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(models[args.model])
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = AutoModelForCTC.from_pretrained(models[args.model], ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id)
    

    @dataclass
    class DataCollatorCTCWithPadding:
    
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

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
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.freeze_feature_extractor()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def add_prefix(example):
        example['audio_filepath'] = os.path.join(args.wav_dir, example['audio_filepath'])
        return example
    
    def prepare_dataset(batch):
        audio = batch["audio_filepath"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"].upper()).input_ids
        return batch

    data_files = {"train":args.train_manifest, "validation": args.val_manifest}
    train_dataset = load_dataset("json", data_files=data_files, split="train")
    train_dataset = train_dataset.map(add_prefix)
    train_dataset = train_dataset.cast_column("audio_filepath", Audio(sampling_rate=16_000))
    train_dataset = train_dataset.map(prepare_dataset, num_proc=4)

    eval_dataset = load_dataset("json", data_files=data_files, split="validation")
    eval_dataset = eval_dataset.map(add_prefix)
    eval_dataset = eval_dataset.cast_column("audio_filepath", Audio(sampling_rate=16_000))
    eval_dataset = eval_dataset.map(prepare_dataset, num_proc=4)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        WER, _, _, _ = helpers.word_error_rate(hypotheses=pred_str, references=label_str)

        return {"wer": WER*100}

    training_args = TrainingArguments(
        output_dir="./test",
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        num_train_epochs=30,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True
        )
    
    trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.feature_extractor
        )
    
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    helpers.print_dict(vars(args))
    main(args)
