import argparse
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer
import torch
from utils import *
import pandas as pd
from datetime import datetime
import os

# Parse the dataset parameter from the console
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True,type=str, help="test dataset to use dataset to use")
args = parser.parse_args()
input_dataset = args.dataset.lower()

# Validate the split parameter
valid_datasets = ["mrpc", "how"]


if input_dataset=="mrpc":
    dataset_test = load_dataset('glue', 'mrpc', split="test")
elif input_dataset=="how":
    #Todo change split to test
    dataset_test = load_dataset('csv', data_files='../../data/HoW/HoW_HuggingFace_format.csv',split="train")
else:
    valid_datasets_str = ", ".join(valid_datasets)
    raise ValueError(f"No dataset found with the name {args.dataset.lower()}. Valid datasets are: {valid_datasets_str}")

# Load dataset and model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

# Load pretrained model and set it to evaluation mode
model.load_state_dict(torch.load('state.pt'), strict=True)
model = model.eval()

# Tokenize dataset
tokenized_dataset = reformat_dataset(dataset_test)

# Setup the trainer
trainer = Trainer(model=model)

# Make predictions on the test dataset
predictions = trainer.predict(tokenized_dataset)

# Create the predictions folder if it does not exist
output_folder = 'predictions'
os.makedirs(output_folder, exist_ok=True)

# Export predictions to a CSV file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"predictions_{args.dataset}-{timestamp}.csv"
output_path = os.path.join(output_folder, output_filename)
df_predictions = pd.DataFrame(predictions.predictions)
df_predictions.to_csv(output_path, index=False)

print(f"Predictions exported to {output_path}")
print(compute_metrics(predictions))