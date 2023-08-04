import torch
import os
from data_loader import Phase, create_data_loader
from utils import parse_arguments_train, instantiate_model, initialize_loss_and_optimizer, run_epoch, set_device

def train_model():
    # Initialize parameters
    args = parse_arguments_train()

    # Load data
    train_dataloader = create_data_loader(data_path=args.data_path, batch_size=args.batch_size, phase=Phase.TRAIN,strategy=args.training_strategy)
    val_dataloader = create_data_loader(data_path=args.data_path, batch_size=args.val_batch_size, phase=Phase.VALIDATION,strategy=args.training_strategy)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}

    # Set device
    device = set_device(args)

    # Create model, loss and optimizer
    model = instantiate_model(args)
    model.to(device)
    loss_function, model_opt= initialize_loss_and_optimizer(model)

    # Run training

    best_loss = float("inf")
    os.makedirs(args.model_save_path, exist_ok=True)
    for epoch in range(args.epochs):
        print(f'\nEPOCH {epoch}')
        for phase in ['train', 'val']:
            is_training= True if phase =='train' else False
            epoch_loss, f1, acc,_,_ = run_epoch(model,device,args.tqdm_print_step, dataloader[phase], loss_function, model_opt,is_training)

            if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
                print("save model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{args.model_save_path}/epoch{epoch}_loss_{epoch_loss}_acc_{f1}_acc_{acc}.pkl')

if __name__ == "__main__":
    train_model()
