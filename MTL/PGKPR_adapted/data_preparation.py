import pandas as pd
import torch
from utils import loadVocab, START_TAG, END_TAG, UNK_TAG, vocab_path
import os
# Import dataset module from CDCR folder
import sys
sys.path.append(r'../../CDCR/data')

from dataset import CrossEncoderDatasetFull, datasets_config_constants

# Modification of the original load_text_con method. Adds start and end tags
def load_text_con(cdcr_dataset_split, max_rows=None):
    if max_rows is None:
        max_rows = len(cdcr_dataset_split.instances)
    
    # Load CDCR data
    inputs = [[START_TAG] + x.strip().split(" ") + [END_TAG] for x in cdcr_dataset_split.instances[0:max_rows]] 
    # I could add this to the CDCR processing. There is unnecessary overhead
    
    labels = cdcr_dataset_split.labels[0:max_rows]

    return inputs, labels

def load_data_tensor_con(cdcr_dataset, max_rows, vocabPath, maxlen=300):
    # maxlen set to 300 until I incorporate more context
    token2idx, _ = loadVocab(vocabPath)

    tokenized_inputs = []
    inputs, labels = load_text_con(cdcr_dataset, max_rows)
    padded_length = maxlen
    for x in inputs:
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]  # Fills with Unknown tag value if token not found
        x = x + [0] * (padded_length - len(x)) 

        tokenized_inputs.append(x)
    return torch.tensor(tokenized_inputs), labels

def save_tensors_with_key(key_index):
    keys = list(datasets_config_constants.keys())
    key = keys[key_index]
    dataset_config = datasets_config_constants[key]

    # Create the 3 splits
    train_data = CrossEncoderDatasetFull(config=dataset_config, split_name='train')
    test_data = CrossEncoderDatasetFull(config=dataset_config, split_name='test')
    val_data = CrossEncoderDatasetFull(config=dataset_config, split_name='val')

    # Load and save the tensors
    train_inputs, train_labels = load_data_tensor_con(cdcr_dataset=train_data, max_rows=None, vocabPath=vocab_path)
    test_inputs, test_labels = load_data_tensor_con(cdcr_dataset=test_data, max_rows=None, vocabPath=vocab_path)
    val_inputs, val_labels = load_data_tensor_con(cdcr_dataset=val_data, max_rows=None, vocabPath=vocab_path)

    folder = "individual_datasets"
    os.makedirs(folder, exist_ok=True)

    torch.save(train_inputs, os.path.join(folder, f'{key}_train_inputs.pt'))
    torch.save(train_labels, os.path.join(folder, f'{key}_train_labels.pt'))

    torch.save(test_inputs, os.path.join(folder, f'{key}_test_inputs.pt'))
    torch.save(test_labels, os.path.join(folder, f'{key}_test_labels.pt'))

    torch.save(val_inputs, os.path.join(folder, f'{key}_val_inputs.pt'))
    torch.save(val_labels, os.path.join(folder, f'{key}_val_labels.pt'))

if __name__ == "__main__":
    print("Available datasets:")
    keys = list(datasets_config_constants.keys())
    for i, key in enumerate(keys):
        print(f"{i}: {key}")
    
    while True:
        try:
            key_index = int(input("Choose the integer corresponding to the dataset you want to process: "))
            if 0 <= key_index < len(keys):
                save_tensors_with_key(key_index)
                print("Tensors saved successfully!")
                break
            else:
                print("Invalid input. Please choose a valid integer from the list above.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
