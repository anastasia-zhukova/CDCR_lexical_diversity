from datasets import load_dataset,DatasetDict
from transformers import AutoModelForSequenceClassification, Trainer
import torch
from utils import *

# Load the new dataset
how_train = DatasetDict.load_from_disk('../../data/HoW/HoW_DatasetDict')["train"]
how_validation = DatasetDict.load_from_disk('../../data/HoW/HoW_DatasetDict')["train"]

# Preprocess the new dataset
how_train_tokenized = reformat_dataset(how_train)
how_validation_tokenized = reformat_dataset(how_validation)

# Load dataset and model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
model.load_state_dict(torch.load('state.pt'), strict=True)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=how_train_tokenized,
    eval_dataset=how_validation_tokenized,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

#Save model state binary
torch.save(model.state_dict(), 'how_finetune_state.pt')