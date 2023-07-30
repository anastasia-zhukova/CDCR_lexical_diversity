import torch
import sys
import os
sys.path.append(r'../../CDCR/data')
from dataset import datasets_config_constants

keys = list(datasets_config_constants.keys())

dict_experiments = {
    0: ['ECBPLUS_ENTITY_CONFIG', 'NIDENT_ENTITY_CONFIG'],
    1: ['ECBPLUS_EVENT_CONFIG', 'GVC_EVENT_CONFIG']
}

# Function to load data and labels for a given split
def load_data_and_labels(dataset, split):
    data_path = f"individual_datasets/{dataset}_{split}_inputs.pt"
    label_path = f"individual_datasets/{dataset}_{split}_labels.pt"

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Data or label file not found for dataset '{dataset}' and split '{split}'.")

    data_inputs, data_labels = torch.load(data_path), torch.load(label_path)
    print(f"source {dataset}, {split}, shape {data_inputs.shape}")
    identifier_tensor = torch.full_like(data_labels, keys.index(dataset))
    return data_inputs, data_labels, identifier_tensor

# List of splits
splits = ["train", "test", "val"]

# Prompt the user to choose a dataset from dict_experiments
while True:
    print("Choose a dataset:")
    for key, datasets in dict_experiments.items():
        print(f"{key}: {datasets}")
    user_choice = input("Enter the dataset number: ")
    try:
        user_choice = int(user_choice)
        if user_choice in dict_experiments:
            selected_dataset = dict_experiments[user_choice]
            break
        else:
            print("Invalid choice. Please enter a valid dataset number.")
    except ValueError:
        print("Invalid input. Please enter a valid integer dataset number.")

# Initialize empty lists for train, test, and val data
train_data = [[] for _ in range(len(splits))]
test_data = [[] for _ in range(len(splits))]
val_data = [[] for _ in range(len(splits))]

# Load data for each dataset and split, and collect them in corresponding lists
for i, dataset in enumerate(selected_dataset):
    for j, split in enumerate(splits):
        data_inputs, data_labels, identifier_tensor = load_data_and_labels(dataset, split)
        if split == "train":
            train_data[j].append((data_inputs, data_labels, identifier_tensor))
        elif split == "test":
            test_data[j].append((data_inputs, data_labels, identifier_tensor))
        else:
            val_data[j].append((data_inputs, data_labels, identifier_tensor))

# Concatenate the lists of tensors into single tensors and save them
output_directory = "./mixed_dataset"
os.makedirs(output_directory, exist_ok=True)

for split, data in zip(splits, [train_data, test_data, val_data]):
    data = [item for sublist in data for item in sublist]  # Flatten the list of tuples

    # Skip saving if the data list is empty
    if len(data) == 0:
        continue

    # Unzip the list of tuples
    inputs, labels, identifiers = zip(*data)

    # Concatenate the tensors
    concatenated_inputs = torch.cat(inputs)
    concatenated_labels = torch.cat(labels)
    concatenated_identifiers = torch.cat(identifiers)

    # Print the shapes before saving
    print(f"{split} - inputs shape: {concatenated_inputs.shape}")
    print(f"{split} - labels shape: {concatenated_labels.shape}")
    print(f"{split} - identifiers shape: {concatenated_identifiers.shape}")

    # Save the concatenated tensors
    torch.save(concatenated_inputs, f"{output_directory}/{split}_inputs.pt")
    torch.save(concatenated_labels, f"{output_directory}/{split}_labels.pt")
    torch.save(concatenated_identifiers, f"{output_directory}/{split}_identifiers.pt")
