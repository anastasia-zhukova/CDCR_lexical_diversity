# Code Description

This code performs prediction using a pre-trained sequence classification model on a specified dataset. It supports two datasets: "mrpc" and "how". The predictions are exported to a CSV file, and metrics are computed based on the predictions.

## Prerequisites

- Python 3.x
- `argparse` library
- `datasets` library
- `transformers` library
- `torch` library
- `pandas` library

## Usage

1. Ensure that the required libraries are installed.
2. Run the script with the following command:

```shell
python 1_make_predictions.py --dataset <dataset_name> --model <model's states file name>
```
Example:

```shell
python 1_make_predictions.py --dataset mrpc --model state.pt
```

# Classification Metrics and Confusion Matrix

This script calculates classification metrics and displays a confusion matrix based on predicted and true labels.

## Usage

To run the script, execute the following command:

```shell
python 2_evaluate_predictions.py --prediction_file prediction_file_name.csv
```


## Dataset

The script uses the following datasets:

- MRPC (Microsoft Research Paraphrase Corpus): The MRPC dataset is loaded using the `load_dataset` function from the "glue" module of the Hugging Face library.
- HOW (headlines of war): The HOW dataset is loaded using the `load_dataset` function from the "csv" module of the Hugging Face library. The dataset file should be located at `../../data/HoW/HoW_HuggingFace_format.csv`.

## Output

The script calculates classification metrics and displays them using the `classification_report` function from scikit-learn. Additionally, it generates a confusion matrix and prints it with labels for the true label and predicted label axes.

Example output:

      0       0.80      0.75      0.77       100
      1       0.82      0.86      0.84       100

accuracy                           0.81       200

Confusion Matrix:
True Label | Predicted Label
0 | 75 25
1 | 14 86