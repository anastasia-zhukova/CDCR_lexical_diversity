import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset, DatasetDict

# Parse the dataset parameter from the console
parser = argparse.ArgumentParser()
parser.add_argument("--prediction_file", required=True,type=str, help="choose a prediction from the predictions folder. e.g.: predictions_how-2023-06-16_11-17-11.csv")
args = parser.parse_args()
input_prediction = args.prediction_file

#Validate prediction_file input
#This part could be simplified but it is not urgent
split_name = input_prediction.split("_")
if len(split_name)<2:
    raise ValueError(f"use a valid prediction name")
dataset_name = split_name[1]

# Validate the split parameter
valid_datasets = ["mrpc", "how"]
valid_datasets_str = ", ".join(valid_datasets)
if dataset_name not in valid_datasets:
    raise ValueError(f"{dataset_name} is not a valid name. Valid datasets are: {valid_datasets_str}")


def print_metrics(y_true, y_pred):
    print(classification_report(y_true=y_true, y_pred=y_pred))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Get the number of classes
    num_classes = len(np.unique(y_true))

    # Print confusion matrix with labels
    print("Confusion Matrix:")
    print("True Label | Predicted Label")
    for i in range(num_classes):
        row = "     " + " ".join(str(count) for count in cm[i])
        print(f"{i:11d} |{row}")


predictions = pd.read_csv(f"predictions/{input_prediction}")
dataset_test=pd.DataFrame()
if dataset_name=="mrpc":
    dataset_test = load_dataset('glue', 'mrpc', split="test")
    ## Import prediction

if dataset_name=="how":
    dataset_test = DatasetDict.load_from_disk('../../data/HoW/HoW_DatasetDict')["test"]
    
predictions = predictions.to_numpy()
predictions = np.argmax(predictions, axis=-1)

actual_values = np.array(dataset_test["label"])
print(predictions.shape,actual_values.shape)
print_metrics(y_true=actual_values, y_pred=predictions)