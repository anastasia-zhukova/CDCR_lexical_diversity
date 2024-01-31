
import torch
from torch.utils.data import DataLoader

def flatten_tokenizer_output(tokenized_mentions):
    # If model is Roberta
    input_ids = [torch.tensor(tokenized_output['input_ids']) for tokenized_output in tokenized_mentions]
    attention_masks = [torch.tensor(tokenized_output['attention_mask']) for tokenized_output in tokenized_mentions]

    # Convert lists to tensors
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, attention_masks

def cl_collate_fn_train(batch):
    # `batch` is a list of samples, where each sample is a tuple of (tokenized_phrases_reshaped, label)
    # Return a tuple of lists, where each list contains the corresponding elements from the samples
    tokenized_mentions1,tokenized_mentions2, labels = zip(*batch)

    input_ids1, attention_masks1 = flatten_tokenizer_output(tokenized_mentions1)
    input_ids2, attention_masks2 = flatten_tokenizer_output(tokenized_mentions2)
    # If model is Roberta

    labels = torch.tensor(labels)

    return input_ids1, attention_masks1,input_ids2,attention_masks2, labels

def cl_collate_fn_val(batch):

    index,tokenized_mentions= zip(*batch)

    input_ids, attention_masks = flatten_tokenizer_output(tokenized_mentions)
    index = torch.tensor(index)

    return index, input_ids, attention_masks


def create_data_loader(cl_cdcr,batch_size):
    shuffle = True if cl_cdcr.mode=='train' else False
    collate_fn = cl_collate_fn_train if mode=='train' else cl_collate_fn_val
    dataloader= DataLoader(cl_cdcr, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn )
    return dataloader

