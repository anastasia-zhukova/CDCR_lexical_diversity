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
    

class MetaDataLoader:
    def __init__(self, data_loaders):
        self.data_loaders = data_loaders
        self.num_loaders = len(data_loaders)
        self.loaders = [iter(loader) for loader in data_loaders.values()]
        self.num_batches_fetched = 0

    def __len__(self):
        return sum(len(loader) for loader in self.data_loaders.values())

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_batches_fetched == len(self):
            # All batches have been fetched, raise StopIteration
            raise StopIteration

        selected_loader_idx = self.num_batches_fetched % self.num_loaders
        try:
            batch = next(self.loaders[selected_loader_idx])
        except StopIteration:
            # Reset the iterator for the selected loader if it's exhausted
            self.loaders[selected_loader_idx] = iter(self.data_loaders[selected_loader_idx])
            batch = next(self.loaders[selected_loader_idx])

        self.num_batches_fetched += 1
        return batch


class RandomMetaDataLoader:
    def __init__(self, data_loaders,identifier_prob_dict):
        self.data_loaders = data_loaders
        self.num_loaders = len(data_loaders)
        self.loaders = [iter(loader) for loader in data_loaders.values()]
        self.num_batches_fetched = 0
        self.identifier_prob_dict = identifier_prob_dict
        

    def __len__(self):
        return sum(len(loader) for loader in self.data_loaders.values())


    def __iter__(self):
        return self

    def __next__(self):
        if self.num_batches_fetched == len(self):
            # All batches have been fetched, raise StopIteration
            raise StopIteration

        # Sample an index using multinomial based on the probabilities
        selected_loader_idx = torch.multinomial(torch.tensor(list(self.identifier_prob_dict.values())), 1).item()

        # Retrieve the identifier associated with the sampled index
        selected_identifier = list(self.identifier_prob_dict.keys())[selected_loader_idx]

        try:
            batch = next(self.loaders[selected_identifier])
        except StopIteration:
            # Reset the iterator for the selected loader if it's exhausted
            self.loaders[selected_identifier] = iter(self.data_loaders[selected_identifier])
            batch = next(self.loaders[selected_identifier])

        self.num_batches_fetched += 1
        return batch
    

def create_data_loader(data_path, batch_size, phase,strategy='sequential'):

    sequential=True
    random_sequence=True

    if strategy=='simultaneous':
        sequential=False
    elif strategy =='sequential':
        random_sequence==False

    phase_str = phase.value

    inputs_path = os.path.join(data_path, f"{phase_str}_inputs.pt")
    label_path = os.path.join(data_path, f"{phase_str}_labels.pt")
    identifier_path = os.path.join(data_path, f"{phase_str}_identifiers.pt")

    data_inputs, data_labels = torch.load(inputs_path), torch.load(label_path)
    identifier_tensor = torch.load(identifier_path)  # Load the identifier tensor


    dataset = TensorDataset(data_inputs, data_labels, identifier_tensor)  # Include the identifier tensor in the dataset
    
    if sequential:

        # If sequential=True, create a dictionary of dataloaders for each unique identifier value
        unique_identifiers, identifier_counts = torch.unique(identifier_tensor, return_counts=True)
        identifier_prob_dict = {identifier.item(): count.item() / len(identifier_tensor) for identifier, count in zip(unique_identifiers, identifier_counts)}
        data_loaders = {}

        for identifier in unique_identifiers:
            indices = torch.nonzero(identifier_tensor == identifier).squeeze()
            subset_dataset = torch.utils.data.Subset(dataset, indices)
            data_loaders[identifier.item()] = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

        if random_sequence:
            return RandomMetaDataLoader(data_loaders, identifier_prob_dict)
        else:
            return MetaDataLoader(data_loaders)

    else:
        if phase_str == Phase.TRAIN:
            sampler = RandomSampler(dataset)
        else:
            sampler = None
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
