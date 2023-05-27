import evaluate
import numpy as np

from transformers import AutoTokenizer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=25, #add info to log after n steps
    save_steps=50, #Save checkpoints after n steps
    save_total_limit=3,
    run_name= "model_run",
    disable_tqdm = False,
    load_best_model_at_end = True,
    metric_for_best_model="eval_accuracy",
    
)

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