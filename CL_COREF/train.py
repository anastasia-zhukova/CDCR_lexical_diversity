import argparse
import numpy as np
import os
import random
from transformers import logging
import torch

from dataloader import create_data_loader
from dataset import CL_Dataset
from evaluation import predict_and_evaluate
from model import CustomContrastiveLoss, create_tokenizer, create_siamese_model, create_optimizer

logging.set_verbosity_error()
# Set random seed for reproducibility
seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Functions

def train_func(model, optimizer, train_loader,loss_fn):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cumulative_loss=0
    model.train()
    for input_ids1, attention_masks1,input_ids2,attention_masks2, labels in train_loader:
        #Load data
        input_ids1, attention_masks1,input_ids2,attention_masks2 = input_ids1.to(device), attention_masks1.to(device),input_ids2.to(device),attention_masks2.to(device)
        labels = labels.to(device)   
        #Zero the gradients
        optimizer.zero_grad() 

        #Compute loss
        output_mentions1 = model(input_ids1, attention_masks1)
        output_mentions2 = model(input_ids2, attention_masks2)
        loss = loss_fn(output_mentions1, output_mentions2, labels)
        loss.backward()
        #Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()
        cumulative_loss += loss.item()
    
    return cumulative_loss

def test_func(model, data_loader,cl_cdcr_dataset,clustering_threshold,metric="conll"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    df_metrics, _ = predict_and_evaluate(dataloader=data_loader,device=device,model=model,cl_cdcr_dataset=cl_cdcr_dataset,metrics="all",clustering_threshold=clustering_threshold)
    return df_metrics.loc[metric,"F1"]



def train_setup(gpu_id,config):

    #Data setup

    ##Load special tokens
    end_start_special_tokens = []
    for token in config["special_tokens"]:
        end_start_special_tokens.append(f"[{token}]")
        end_start_special_tokens.append(f"[/{token}]")

    tokenizer = create_tokenizer(config["language_model"],end_start_special_tokens)

    ##create dataloaders
    cl_cdcr_train = CL_Dataset(name=config["train_dataset_name"],data_paths=config["train_conll_paths"],context_method=config["context_method"],
                        mention_type=config["mention_type"],tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="train",
                        additional_sampling=config["additional_sampling"]) 

    cl_cdcr_val = CL_Dataset(name=config["val_dataset_name"],data_paths=config["val_conll_paths"],context_method=config["context_method"],
                    mention_type=config["mention_type"],tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="inference") 
 
    train_dataloader = create_data_loader(cl_cdcr=cl_cdcr_train,batch_size=config["train_batch_size"])

    val_dataloader = create_data_loader(cl_cdcr=cl_cdcr_val,batch_size=config["val_batch_size"])

    ##Model creation
    model = create_siamese_model(config["language_model"],tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ##Optimizer and loss
    
    optimizer = create_optimizer(config["learning_rate"],model)
    contrastive_loss = CustomContrastiveLoss(margin=config["margin"])

    ##Training loop
    best_metric=0.0
    for epoch in range(args.num_epochs):
        cumulative_loss = train_func(model=model, optimizer=optimizer, train_loader=train_dataloader,loss_fn=contrastive_loss)
        print(f'epoch {epoch+1}/{args.num_epochs}, Train accumulated loss: {round(cumulative_loss,2)}')
        epoch_metric = test_func(model=model, data_loader=val_dataloader,cl_cdcr_dataset=cl_cdcr_val,clustering_threshold=config["clustering_threshold"],metric=config["metric"])
        
        # Store the best model
        if epoch_metric>best_metric:
            best_metric=epoch_metric
            torch.save(model.state_dict(), config["model_path"])

            print(f'epoch {epoch+1}/{args.num_epochs} obtained better {config["metric"]}, model saved')  
            print("-------------------------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument("--data_setup",type=int,default=0)
    parser.add_argument('--mention_type', type=str, default='event')
    parser.add_argument('--context_method', type=str, default='hsu')
    parser.add_argument('--language_model', type=str, default='roberta-large')
    parser.add_argument('--special_tokens', type=str, nargs='+', default=["E"])
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-7, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='margin')
    parser.add_argument('--clustering_threshold', type=float, default=0.2, help='clustering threshold')
    parser.add_argument('--additional_sampling', action='store_true', default=False, help='Additional sampling or not')
    parser.add_argument('--metric', type=str, default='conll')
    #Job ID to check logs
    parser.add_argument('--job_id', type=str, default= '')
    # Number of GPUs available
    parser.add_argument('--world_size', default=1, type=int, 
                        help='number of gpus for distributed training')
    
    # Parse arguments
    args = parser.parse_args()
    
    #Create model path
    out_dir = os.path.join(os.getcwd(),"models",args.mention_type,args.language_model,f"tokens_{'_'.join(args.special_tokens)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(out_dir, f"siamese_model_batch_{args.train_batch_size}_epochs_{args.num_epochs}_margin_{args.margin}_threshold_{args.clustering_threshold}_setup_{args.data_setup}.pth")
    print("model path:", model_path)
    print("training metrics:", args.metric)

    #Create data source paths
    base_data_path = os.path.abspath('../CDCR/data')
    
    if args.data_setup==0:
        train_dataset_name = "ecbplus_train"
        train_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'train')]

        val_dataset_name = "ecbplus_val"
        val_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'val')]
    elif args.data_setup==1:
        train_dataset_name = "ecbplus_fcc_train"
        train_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'train'),
                             os.path.join(base_data_path,"FCC-prep","output_data_FCC-T", 'train')]

        val_dataset_name = "ecbplus_fcc_train"
        val_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'val'),
                           os.path.join(base_data_path,"FCC-prep","output_data_FCC-T", 'val')]
    elif args.data_setup==2:
        train_dataset_name = "ecbplus_np4e_train"
        train_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'train'),
                             os.path.join(base_data_path,"NP4E-prep","output_data", 'train')]

        val_dataset_name = "ecbplus_np4e_val"
        val_conll_paths = [os.path.join(base_data_path,"ECBplus-prep","output_data", 'val'),
                           os.path.join(base_data_path,"NP4E-prep","output_data", 'val')]
    else:
        raise Exception(f"data_setup={args.data_setup} is not a valid option")

    if args.additional_sampling:
        print("Additional sampling activated")
    else:
        print("No additional sampling activated")

    args_dict = vars(args)
    args_dict["train_dataset_name"] = train_dataset_name
    args_dict["train_conll_paths"] = train_conll_paths
    args_dict["val_dataset_name"] = val_dataset_name
    args_dict["val_conll_paths"] = val_conll_paths

    args_dict["model_path"] = model_path

    if args.world_size>1:

        #Parallel processing
        ngpus_per_node = torch.cuda.device_count()
        print(f"there are {ngpus_per_node} GPUs available")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501" 
        torch.multiprocessing.spawn(fn=train_setup,args=(args_dict,), nprocs = 2)
    else:
        train_setup(0,args_dict)