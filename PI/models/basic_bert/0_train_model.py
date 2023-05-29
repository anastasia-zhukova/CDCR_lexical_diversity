#import libraries
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer
import torch
from utils import *

#load dataset and model
dataset = load_dataset('glue', 'mrpc')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

# tokenize dataset
tokenized_dataset = reformat_dataset(dataset)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

#Save model state binary
torch.save(model.state_dict(), 'state.pt')
torch.save(model, 'model.pt')