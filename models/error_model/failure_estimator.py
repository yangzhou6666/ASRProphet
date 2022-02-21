import numpy as np
from scipy.special import softmax
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import torch

class Estimator:
    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    def setName(self, name: str):
        self.name = name

    def fit(self, X: [str], y: [int]):
        raise NotImplementedError()

    def predict(self, X: [str]):
        raise NotImplementedError()


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HuggingFaceTransformer(Estimator):
    def __init__(self, name, output_dir, logging_dir, num_labels=2, max_sequence_length=128):
        Estimator.__init__(self, name=name)

        ## init boiler plate
        self.output_dir = output_dir
        self.logging_dir = logging_dir

        ## init model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels)

        ## init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.max_sequence_length = max_sequence_length

    def fit(self, X: [str], y: [int]):

        self.model.to(self.device)
        self.model.train()

        train_texts = X
        train_labels = y

        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        train_dataset = HuggingFaceDataset(train_encodings, train_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,          # output directory
            num_train_epochs=5,              # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            learning_rate=5e-05,
            weight_decay=0.01,               # strength of weight decay
            logging_dir=self.logging_dir,            # directory for storing logs
            logging_steps=10,
            load_best_model_at_end=True
        )

        self.trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=self.model,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset         # training dataset
        )

        self.trainer.train()

        self.tokenizer.save_pretrained(self.output_dir)
        self.trainer.save_model(self.output_dir)

    def predict(self, X: [str]):
        self.model.to(self.device)
        self.model.eval()

        test_texts = X
        test_labels = [0] * len(X)

        test_encodings = self.tokenizer(
            test_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        test_dataset = HuggingFaceDataset(test_encodings, test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False)

        res = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels)
            preds = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
            res.extend(preds)
        res = np.array(res)

        ## assume that 0 is the label for succesful
        ## and 1 is the label for failed
        # succesfull_probability = res[:, 0]
        failed_probability = res[:, 1]

        del test_loader
        torch.cuda.empty_cache()

        return failed_probability
