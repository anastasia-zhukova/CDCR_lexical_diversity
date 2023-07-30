from tqdm import tqdm
from architecture import Embeddings, PositionalEncoding, Encoder, Batch,BinaryClassifier, PGKPR
import torch.nn as nn
import torch
import math
from enum import Enum
from utils import loadVocab, vocab_path

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#Libraries for multi-gpu processing:
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import socket
# Define Enum for phases
class Phase(Enum):
    TRAIN = 'train'
    VALIDATION = 'val'

# Function to load and preprocess data
def load_preprocess_data(data_path, label_path, batch_size, phase):
    data_inputs, data_labels = torch.load(data_path), torch.load(label_path)
    dataset = TensorDataset(data_inputs, data_labels)
    if phase == Phase.TRAIN:
        sampler = RandomSampler(dataset)
    else:
        sampler = None
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size,pin_memory=True)

# Load data
train_dataloader = load_preprocess_data("train_inputs.pt", "train_labels.pt", batch_size=64, phase=Phase.TRAIN)
val_dataloader = load_preprocess_data("val_inputs.pt", "val_labels.pt", batch_size=64, phase=Phase.VALIDATION)


#Load tokenizer
LOWERCASE = True
token2idx, idx2token = loadVocab(vocab_path, lowercase=LOWERCASE) 
vocab_size = len(token2idx)

#Define model parameters
D_MODEL = 512

MID_LOSS = True
DOUBLE_POSITION = False
tqdm_print_step = 1
model_save_path = 'models/'
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

# Instantiate your model
def create_model():
    model = PGKPR(d_model=D_MODEL, vocab=vocab_size, mid_loss=MID_LOSS, double_pe=DOUBLE_POSITION, shared_emb=True)
    device = torch.device('cuda:0')  # Set the default GPU device
    model.to(device)  # Move the model's parameters to the default GPU
    return model

def validate_single_gpu(gpu_id, model_state_dict, val_loader, loss_function):
    # Instantiate a new model independent of the one used for training
    model = create_model()
    # Create a new state_dict without the 'module.' prefix added by parallel module
    new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    # Load the model state dictionary of the training model
    model.load_state_dict(new_state_dict)
    device = torch.device(f'cuda:{gpu_id}')
    model.to(device)

    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for i, data in enumerate(val_loader):
                src, tgt = data
                src, tgt = src.to(device), tgt.to(device)
                batch = Batch(src, tgt)

                binary_loss = run_once(model, batch, loss_function, optimizer=None, is_training=False)
                total_loss += binary_loss.item()

                if i % tqdm_print_step == 0:
                    pbar.update(tqdm_print_step)
                    pbar.set_description(f'Validation loss: {binary_loss:.4f}')

    epoch_loss = total_loss / len(val_loader)
    print(f"Validation loss: {epoch_loss:.4f}")

    return epoch_loss



#Run training
def run_once(model, batch,loss_function,optimizer,is_training):
    predicted_probs= model.forward(src=batch.src,src_mask=batch.src_mask)
    predicted_probs = predicted_probs.squeeze()

    loss = loss_function(predicted_probs, batch.trg)
    
    if is_training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

# Define a function for distributed training (DataParallel)
def train_multi_gpu(gpu_id):
    os.environ['RANK'] = str(gpu_id) 
    
    # Set up distributed training
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the distributed backend (required for DataParallel)
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set up the device for the primary GPU
    # device = torch.device(f'cuda:{gpu_id}')
    device = torch.device('cuda:0')
    # Create model and optimizer
    model = create_model()
    model = DataParallel(model, device_ids=[i for i in range(world_size)])
    model.to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # Load data for multi-GPU training
    train_dataloader = load_preprocess_data("train_inputs.pt", "train_labels.pt", batch_size=BATCH_SIZE, phase=Phase.TRAIN)
    val_dataloader = load_preprocess_data("val_inputs.pt", "val_labels.pt", batch_size=VAL_BATCH_SIZE, phase=Phase.VALIDATION)
    
    #Early stop parameters:
    early_stop_patience = 3  # Number of epochs to wait for improvement
    early_stop_counter = 0
    best_loss = float("inf")
    
    # Train the model on a single or multiple GPUs
    if world_size == 1:
        # Training on a single GPU
        print("Processed stop. Only one GPU available")

    else:
        # Training on multiple GPUs using DataParallel
        for epoch in range(5):  # TODO: Define epochs parameter from a config file
            print(f'\nEPOCH {epoch}')
            model.train()
            total_loss = 0

            with tqdm(total=len(train_dataloader)) as pbar:
                for i, data in enumerate(train_dataloader):
                    src, tgt = data
                    src, tgt = src.to(device), tgt.to(device)
                    batch = Batch(src, tgt)

                    binary_loss = run_once(model, batch, loss_function, optimizer, is_training=True)
                    total_loss += binary_loss.item()

                    if i % tqdm_print_step == 0:
                        pbar.update(tqdm_print_step)
                        pbar.set_description(f'train loss: {binary_loss:.4f}')

            epoch_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch}, train loss: {epoch_loss:.4f}")
            val_loss = validate_single_gpu(gpu_id, model.state_dict(), val_dataloader, loss_function)
            print(f"Epoch {epoch}, validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                print("Saving model")
                torch.save(model.module.state_dict(), f'{model_save_path}/epoch{epoch}_loss_{val_loss:.4f}.pkl')
            else:
                early_stop_counter += 1

            # Check if early stopping criterion is met
            if early_stop_counter >= early_stop_patience:
                print("Early stopping: Performance on validation dataset degraded.")
                break

# Main entry point for distributed training
if __name__ == '__main__':
    #https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904    
    # Get the local IP address using the socket library
    local_ip = socket.gethostbyname(socket.gethostname())

    os.environ['MASTER_ADDR'] = local_ip  # Set the MASTER_ADDR using SLURM_LAUNCH_NODE_IPADDR
    os.environ['MASTER_PORT'] = '12340'  # Replace for another 5-digit port if busy
    os.environ['WORLD_SIZE'] = '2' # Assuming SLURM -G A100:2, use 2 GPUs for training
    mp.spawn(train_multi_gpu, nprocs=2)  # Assuming SLURM -G A100:2, use 2 GPUs for training