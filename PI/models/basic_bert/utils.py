import evaluate
import numpy as np

#Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")


def reformat_dataset(dataset):
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda rows: {'labels': rows['label']}, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return tokenized_dataset

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels,_ = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)