from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
import torch
import evaluate
import numpy as np
from transformers import Trainer

model_save_path = "./fine_tuned_bert_hf_lib"
checkpoint = "bert-base-uncased"


def train_and_evaluate_model(model, tokenizer, if_train=False):
    raw_datasets = load_dataset("glue", "mrpc")

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if if_train:
        trainer.train()

    metric = evaluate.load("glue", "mrpc")
    predictions = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(predictions.predictions, axis=-1)
    print(metric.compute(predictions=preds, references=predictions.label_ids))

    torch.save(model, model_save_path)
    return model


if __name__ == "__main__":
    # model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = torch.load(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_and_evaluate_model(model, tokenizer)
