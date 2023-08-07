import torch

from data_loader import Phase, create_data_loader
from utils import parse_arguments_eval, instantiate_model, run_epoch, set_device, save_predictions_to_csv, extract_dmodel_from_model_name

def evaluate_model():
    # Initialize parameters
    args = parse_arguments_eval() #This has to differ from train 

    test_dataloader = create_data_loader(data_path=args.data_path, batch_size=args.batch_size, phase=Phase.TEST,strategy='simultaneous')

    # Set device
    device = set_device(args)

    # I need to load a model and load its weights
    # Load the model state dictionary from a file
    model_path = f'{args.model_save_path}/{args.model_name}.pkl'
    #Extract d_model from the model's name
    args.d_model = extract_dmodel_from_model_name(model_path)
    model = instantiate_model(args)

    loaded_model_state_dict = torch.load(model_path)

    # Create a new state_dict without the 'module.' prefix added by parallel module

    
    adjusted_state_dict = {k.replace('module.', ''): v for k, v in loaded_model_state_dict.items()}
    model.load_state_dict(adjusted_state_dict)
    model.to(device)

    _, f1, acc, y_true, y_pred = run_epoch(model,device,args.tqdm_print_step, test_dataloader, loss_function= None,optimizer=None,scheduler=None,phase= "test")
    y_pred_binary = (y_pred> 0.5).astype(int)
    model_name_with_metrics = f"{args.model_name},acc_{acc}_f1{f1}"
    save_predictions_to_csv(y_true, y_pred_binary,y_pred, "predictions",model_name_with_metrics)

if __name__ == "__main__":
    evaluate_model()
