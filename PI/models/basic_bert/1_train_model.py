#import libraries
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, Trainer
import torch
from utils import *

#Load dataset, model and tokenizer
dataset_test = load_dataset('glue', 'mrpc', split='test')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Load pretrained model and set it to evaluation mode
model.load_state_dict(torch.load('state.pt'), strict=True)
model = model.eval()

# setup the trainer
trainer = Trainer(
    model=model
)

# tokenize dataset
tokenized_dataset = reformat_dataset(dataset_test)

# Make predictions on the test dataset
predictions = trainer.predict(tokenized_dataset)

print(compute_metrics(predictions))