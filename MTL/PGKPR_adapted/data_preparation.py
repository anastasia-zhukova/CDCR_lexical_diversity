import pandas as pd
import torch
from utils import START_TAG,SEP_TAG, END_TAG, UNK_TAG, add_pi_datasets_to_config_dict
import os
from datasets import DatasetDict, load_dataset #Huggingface library
import traceback

from tokenizer import tokenizer 
# Import dataset module from CDCR folder
import sys
sys.path.append(r'../../CDCR/data')

from dataset import CrossEncoderDatasetFull, datasets_config_constants


#Add paraphrase identification datasets:
datasets_config_constants_with_pi = add_pi_datasets_to_config_dict(datasets_config_constants)

def encode_in_small_batches(full_batch):
    # Define the batch size for encoding (adjust as needed)
    batch_size = 10000

    # Initialize an empty list to store encoded tensors
    encoded_ids = []

    # Loop through the big batch in smaller chunks
    for i in range(0, len(full_batch), batch_size):
        print(f"encoding_subatch {i/batch_size} out of {len(full_batch)/batch_size}")
        # Get the current batch of sentences
        current_batch = full_batch[i:i + batch_size]

        # Encode the current batch
        current_encoded_batch = tokenizer.encode_batch(current_batch)
        current_encoded_batch_ids = [encoded.ids for encoded in current_encoded_batch]
        # Append the encoded tensors to the list
        encoded_ids.extend(current_encoded_batch_ids)
    
    return torch.tensor(encoded_ids)
    
def tokenize_inputs(dataset_split,task, max_rows=None):

    if task=="CDCR":
        if max_rows is None:
            max_rows = len(dataset_split.instances)
        
        # Get the input_ids from each Encoding object and stack them as a tensor
        input_ids_tensor = encode_in_small_batches(dataset_split.instances[0:max_rows])
        labels = dataset_split.labels[0:max_rows]

    elif task=="PI":
        paired_sentences = [(sent1, sent2) for sent1, sent2 in zip(dataset_split["sentence1"], dataset_split["sentence2"])]
        
        if max_rows is None:
            max_rows = len(paired_sentences)

        input_ids_tensor = encode_in_small_batches(paired_sentences[0:max_rows])
        labels = dataset_split["label"][0:max_rows]
        labels = torch.Tensor(labels)
    print("object type of inputs",type(input_ids_tensor))
    print("object type of labels",type(labels))
    return input_ids_tensor, labels



def save_tensors_with_key(key_index):
    keys = list(datasets_config_constants_with_pi.keys())
    key = keys[key_index]
    dataset_config = datasets_config_constants_with_pi[key]
    #HoW processing:
    if key in ["HOW","MRPC"]:
        
        if key=="HOW":
            dataset_dict = DatasetDict.load_from_disk(dataset_config)
        else:
            dataset_dict = load_dataset('glue', dataset_config)

        train_data= dataset_dict["train"]
        test_data= dataset_dict["test"]
        val_data= dataset_dict["validation"]

        task="PI"
    else:

        # Create the 3 splits
        train_data = CrossEncoderDatasetFull(config=dataset_config, split_name='train')
        test_data = CrossEncoderDatasetFull(config=dataset_config, split_name='test')
        val_data = CrossEncoderDatasetFull(config=dataset_config, split_name='val')

        task="CDCR"

    # Load and save the tensors
    print("processing train")
    train_inputs, train_labels = tokenize_inputs(dataset_split=train_data,task=task, max_rows=None)
    print("processing test")
    test_inputs, test_labels = tokenize_inputs(dataset_split=test_data,task=task, max_rows=None)
    print("processing validation")
    val_inputs, val_labels = tokenize_inputs(dataset_split=val_data, task=task,max_rows=None)

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

    keys = list(datasets_config_constants_with_pi.keys())
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
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        except ValueError:
            print("Invalid input. Please enter an integer.")

