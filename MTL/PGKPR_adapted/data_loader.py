from enum import Enum
import torch
import os
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


# Define Enum for phases
class Phase(Enum):
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'




class Batch:
    "Object for holding a batch of data with mask during training."
    #I modified it from the original because I won't use another sequence but a binary label
    def __init__(self, src, trg,dataset_id, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
        self.dataset_id = dataset_id
    


def create_data_loaders(data_path, batch_size, phase):
    phase_str = phase.value
    inputs_path = os.path.join(data_path, f"{phase_str}_inputs.pt")
    label_path = os.path.join(data_path, f"{phase_str}_labels.pt")
    identifier_path = os.path.join(data_path, f"{phase_str}_identifiers.pt")

    data_inputs, data_labels = torch.load(inputs_path), torch.load(label_path)
    identifier_tensor = torch.load(identifier_path)  # Load the identifier tensor
    dataset = TensorDataset(data_inputs, data_labels, identifier_tensor)  # Include the identifier tensor in the dataset
    if phase_str == Phase.TRAIN:
        sampler = RandomSampler(dataset)
    else:
        sampler = None
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
