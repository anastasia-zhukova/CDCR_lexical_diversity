import torch
from data_loader import Phase, create_data_loaders
from utils import parse_arguments, instantiate_model, run_epoch, initialize_loss_and_optimizer

#Libraries for multi-gpu processing:
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import socket


def validate_single_gpu(args,gpu_id, model_state_dict, val_loader, loss_function):
    # Instantiate a new model independent of the one used for training
    model = instantiate_model(args)
    device = torch.device('cuda:0')
    model.to(device)
    # Create a new state_dict without the 'module.' prefix added by parallel module
    new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

    # Load the model state dictionary of the training model
    model.load_state_dict(new_state_dict)
    device = torch.device(f'cuda:{gpu_id}')
    model.to(device)

    epoch_loss = run_epoch(model,device,args.tqdm_print_step, val_loader, loss_function, None,is_training=False)


    print(f"Validation loss: {epoch_loss:.4f}")

    return epoch_loss


# Define a function for distributed training (DataParallel)
def train_multi_gpu(gpu_id):
    args = parse_arguments()
    os.environ['RANK'] = str(gpu_id) 
    
    # Set up distributed training
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the distributed backend (required for DataParallel)
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set up the device for the primary GPU
    # device = torch.device(f'cuda:{gpu_id}')
    device = torch.device('cuda:0')
    # Create model and optimizer
    model = instantiate_model(args)
    model.to(device)

    #Move to parallel
    model = DataParallel(model, device_ids=[i for i in range(world_size)])
    model.to(device)

    loss_function,optimizer =  initialize_loss_and_optimizer(model)

    # Load data
    data_path = "mixed_dataset/"
    train_dataloader = create_data_loaders(data_path=data_path, batch_size=args.batch_size, phase=Phase.TRAIN)
    val_dataloader = create_data_loaders(data_path=data_path, batch_size=args.val_batch_size, phase=Phase.VALIDATION)

    
    #Early stop parameters:
    early_stop_patience = 3  # Number of epochs to wait for improvement
    early_stop_counter = 0
    best_loss = float("inf")
    
    # Train the model on a single or multiple GPUs
    if world_size == 1:
        # Training on a single GPU
        print("Process stopped. Only one GPU available")

    else:
        # Training on multiple GPUs using DataParallel
        for epoch in range(args.epochs): 
            print(f'\nEPOCH {epoch}')
            
            epoch_loss = run_epoch(model,device,args.tqdm_print_step, train_dataloader, loss_function, optimizer,is_training=True)
            print(f"Epoch {epoch}, train loss: {epoch_loss:.4f}")
            
            #Instead of alternating the model in train and val, the validation in this code creates a new model and loads the state_dict
            #To avoid problems finding in which GPU the model is
            val_loss = validate_single_gpu(args,gpu_id, model.state_dict(), val_dataloader, loss_function)
            print(f"Epoch {epoch}, validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                print("Saving model")
                torch.save(model.module.state_dict(), f'{args.model_save_path}/epoch{epoch}_loss_{val_loss:.4f}.pkl')
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