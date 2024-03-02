import argparse
import numpy as np
import os
import random
from ray import train, tune
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

def train_func(model, optimizer, train_loader,loss_fn): #What happens in an epoch, also pass contrastive loss. They are not accumulating loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def test_func(model, data_loader,cl_cdcr_dataset,clustering_threshold):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    df_metrics, _ = predict_and_evaluate(dataloader=data_loader,device=device,model=model,cl_cdcr_dataset=cl_cdcr_dataset,metrics="all",clustering_threshold=clustering_threshold)

    val_conll_f1 = df_metrics.loc["conll","F1"]
    val_lea = df_metrics.loc["lea","F1"]
    val_pd_lea = df_metrics.loc["pd_lea","F1"]
    print(f' eval conll: {round(val_conll_f1,2)},eval lea: {round(val_lea,2)},eval pd_lea: {round(val_pd_lea,2)}')
    return val_pd_lea

def train_setup(config):

    #Data setup

    #Load special tokens
    end_start_special_tokens = []
    for token in config["special_tokens"]:
        end_start_special_tokens.append(f"[{token}]")
        end_start_special_tokens.append(f"[/{token}]")
    #Create tokenizer
    tokenizer = create_tokenizer(config["language_model_name"],end_start_special_tokens)

    #define data paths
    if config["data_setup"]==0:
        train_dataset_name = "ecbplus_train"
        train_conll_paths = ['../CDCR/data/ECBplus-prep/output_data/train']

        val_dataset_name = "ecbplus_val"
        val_conll_paths = ['../CDCR/data/ECBplus-prep/output_data/val']
    else:
        raise Exception(f"data_setup={config["data_setup"]} is not a valid option")

    #create dataloaders

    cl_cdcr_train = CL_Dataset(name=train_dataset_name,data_paths=train_conll_paths,context_method=config["context_method"],
                        mention_type=config["mention_type"],tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="train",
                        export_samples=config["export_samples"], additional_sampling=config["additional_sampling"]) 

    cl_cdcr_val = CL_Dataset(name=val_dataset_name,data_paths=val_conll_paths,context_method=config["context_method"],
                    mention_type=config["mention_type"],tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="inference") 
 
    train_dataloader = create_data_loader(cl_cdcr=cl_cdcr_train,batch_size=config["train_batch_size"])

    val_dataloader = create_data_loader(cl_cdcr=cl_cdcr_val,batch_size=config["val_batch_size"])

    #Model creation
    model = create_siamese_model(config["language_model_name"],tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Optimizer and loss
    
    optimizer = create_optimizer(config["learning_rate"],model)
    contrastive_loss = CustomContrastiveLoss(margin=config["margin"])

    #Training loop
    for i in range(10):
        train_func(model=model, optimizer=optimizer, train_loader=train_dataloader,loss_fn=contrastive_loss)
        pd_lea = test_func(model=model, data_loader=val_dataloader,cl_cdcr_dataset=cl_cdcr_val,clustering_threshold=config["clustering_threshold"])
        
        # Send the current training result back to Tune
        train.report({"pd_lea": pd_lea})

        if i % 5 == 0: #TODO modify this part
            # This saves the model to the trial directory
            torch.save(model.state_dict(), config["model_path"])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument("--data_setup",type=str,default=0)
    parser.add_argument('--mention_type', type=str, default='event')
    parser.add_argument('--context_method', type=str, default='hsu')
    parser.add_argument('--language_model', type=str, default='roberta-large')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--special_tokens', type=str, nargs='+', default=["E"])
    parser.add_argument('--clustering_threshold', type=float, default=0.2, help='clustering threshold')
    parser.add_argument('--export_samples', action='store_true', default=False, help='Export samples or not')
    parser.add_argument('--additional_sampling', action='store_true', default=False, help='Additional sampling or not')
    #Job ID to check logs
    parser.add_argument('--job_id', type=str, default= '')
    # Number of GPUs available
    parser.add_argument('--world_size', default=1, type=int, 
                        help='number of gpus for distributed training')
    
    # Parse arguments
    args = parser.parse_args()
    world_size = args.world_size

    #Create model path
    # out_dir = os.path.join(os.getcwd(),"models",args.mention_type,args.language_model,f"tokens_{'_'.join(args.special_tokens)}")
    out_dir = os.path.join(os.getcwd(),"models","tuning",args.mention_type,args.language_model,f"tokens_{'_'.join(args.special_tokens)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(out_dir, f"siamese_model_batch_hyperparameter_tuning.pth")
    
    #Print general info about the run
    print(f"model_path: {model_path}")
    if args.additional_sampling:
        print("Additional sampling activated")
    else:
        print("No additional sampling activated")

    search_space = {
        #constant
        "data_setup":args.data_setup,
        "context_method": args.context_method,
        "mention_type" : args.mention_type,
        "export_samples" : args.export_samples,
        "language_model_name" : args.language_model,
        "special_tokens": args.special_tokens,
        "additional_sampling" : args.additional_sampling,
        "val_batch_size":args.val_batch_size,
        "model_path":model_path,
        #Variables
        "learning_rate": tune.choice([2e-7, 2e-6, 2e-8]), 
        "margin":tune.choice([0.4, 0.5, 0.6,0.7]),
        "train_batch_size":tune.choice([16,32,64]),
        "clustering_threshold":tune.choice([0.15,0.20,0.25,0.30])
    }

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    tuner = tune.Tuner(
        train_setup,
        param_space=search_space,
    )
    results = tuner.fit()
    df_results = results.get_dataframe()
    print("df_results:")
    print(df_results)
    df_results.to_csv("df_hyperparameter_tuning.csv")
    print("testing the other way to show results")
    dfs = {result.path: result.metrics_dataframe for result in results}
    print(dfs)
