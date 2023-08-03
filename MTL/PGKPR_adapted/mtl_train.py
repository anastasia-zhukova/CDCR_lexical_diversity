import torch

from data_loader import Phase, create_data_loader
from utils import parse_arguments, instantiate_model, initialize_loss_and_optimizer, run_epoch

def train_model():
    # Initialize parameters
    args = parse_arguments()

    # Load data
    data_path = "mixed_dataset/"
    training_strategy = args.training_strategy
    
    train_dataloader = create_data_loader(data_path=data_path, batch_size=args.batch_size, phase=Phase.TRAIN,strategy=training_strategy)
    val_dataloader = create_data_loader(data_path=data_path, batch_size=args.val_batch_size, phase=Phase.VALIDATION,strategy=training_strategy)

    dataloader = {'train': train_dataloader, 'val': val_dataloader}

    # Search device
     # Set the device based on the 'use_cpu' argument
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("use_cpu set to False. Running on", device)

    # Create model, loss and optimizer
    model = instantiate_model(args)
    model.to(device)
    loss_function, model_opt= initialize_loss_and_optimizer(model)

    # Run training

    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f'\nEPOCH {epoch}')
        for phase in ['train', 'val']:
            is_training= True if phase =='train' else False
            epoch_loss = run_epoch(model,device,args.tqdm_print_step, dataloader[phase], loss_function, model_opt,is_training)
            
            print(f"epoch loss: {epoch_loss}")

            if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
                print("save model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{args.model_save_path}/epoch{epoch}_loss_{epoch_loss}.pkl')

if __name__ == "__main__":
    train_model()
