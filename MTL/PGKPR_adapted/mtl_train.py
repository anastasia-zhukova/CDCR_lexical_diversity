import torch
import os
from data_loader import Phase, create_data_loader
from utils import parse_arguments_train, instantiate_model, initialize_loss_optimizer_scheduler, run_epoch, set_device
import datetime

def train_model():
    # Initialize parameters
    args = parse_arguments_train()

    # Load data
    train_dataloader = create_data_loader(data_path=args.data_path, batch_size=args.batch_size, phase=Phase.TRAIN,strategy=args.training_strategy)
    val_dataloader = create_data_loader(data_path=args.data_path, batch_size=args.val_batch_size, phase=Phase.VALIDATION,strategy=args.training_strategy)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}


 
    # Set device
    device = set_device(args)

    #Set weights for imbalance:
    pos_weight = torch.tensor([5.0])
    pos_weight = pos_weight.to(device)

    # Create model, loss and optimizer
    model = instantiate_model(args)
    model.to(device)
    loss_function, model_opt,scheduler= initialize_loss_optimizer_scheduler(model,pos_weight)

    # Run training

    best_loss = float("inf")
    os.makedirs(args.model_save_path, exist_ok=True)
    for epoch in range(args.epochs):
        print(f'\nEPOCH {epoch}')
        for phase in ['train', 'val']:
            
            epoch_loss, f1, acc,_,_ = run_epoch(model,device,args.tqdm_print_step, dataloader[phase], loss_function, model_opt,scheduler,phase)

            if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
                print("save model")
                best_loss = epoch_loss                

                # Get the current date
                current_date = datetime.date.today()

                # Format the time-stamp as a string
                time_stamp = current_date.strftime("%Y-%m-%d")

                model_name =  f'{args.model_save_path}mtl_model_{time_stamp}_dmodel_{args.d_model}_experiment_XX_epoch{epoch}_of_{args.epochs-1}.pkl'
                torch.save(model.state_dict(),model_name)

if __name__ == "__main__":
    train_model()
