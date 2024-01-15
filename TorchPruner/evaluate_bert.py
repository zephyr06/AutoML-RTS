from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from evaluate import evaluator
from datasets import load_dataset


def evaluate_bert_based_model(model, tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    token_clf = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0)
    eval_dataset = load_dataset("conll2003", split="validation[:100]")
    task_evaluator = evaluator("token-classification")

    # run baseline
    results = task_evaluator.compute(
        model_or_pipeline=token_clf,
        data=eval_dataset,
        metric="seqeval",
    )
    print(f"Overall f1 score for our model is {results['overall_f1'] * 100:.2f}%")
    print(f"The avg. Latency of the model is {results['latency_in_seconds'] * 1000:.2f}ms")
    return results['overall_f1'], results['latency_in_seconds'] * 1000

# Model Repository on huggingface.co
# model_id = "dslim/bert-large-NER"
# # Load Model and Tokenizer
# model = AutoModelForTokenClassification.from_pretrained(model_id)
# evaluate_bert_based_model(model)
