import argparse
import numpy as np
import pandas as pd
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

def test_func(model, data_loader,cl_cdcr_dataset,clustering_threshold):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    df_metrics, _ = predict_and_evaluate(dataloader=data_loader,device=device,model=model,cl_cdcr_dataset=cl_cdcr_dataset,metrics="all",clustering_threshold=clustering_threshold,without_singletons=True)
    return df_metrics

def train_hyperparam_tuning(config):

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
    epochs_without_improvement=0

    best_conll = 0
    best_lea = 0
    best_pd_lea = 0
    best_model_state = dict()

    for epoch in range(config["max_epochs"]):
        cumulative_loss = train_func(model=model, optimizer=optimizer, train_loader=train_dataloader,loss_fn=contrastive_loss)
        print(f'epoch {epoch+1}, Train accumulated loss: {round(cumulative_loss,2)}')
        df_metrics= test_func(model=model, data_loader=val_dataloader,cl_cdcr_dataset=cl_cdcr_val,clustering_threshold=config["clustering_threshold"])
        
        optimization_metric = config["optimization_metric"]
        df_metrics.loc[optimization_metric,"F1"]
        #Here I have to do the early stopping
        # Store the best model. Save all best metrics
        if optimization_metric>best_metric:
            best_metric=optimization_metric
            best_conll = df_metrics.loc["conll","F1"]
            best_lea = df_metrics.loc["lea","F1"]
            best_pd_lea = df_metrics.loc["pd_lea","F1"]
            best_model_state= model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement+=1 
        if epochs_without_improvement>=config["patience"]:
            return best_model_state, best_conll,best_lea,best_pd_lea



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mention_type', type=str, default='event')
    parser.add_argument('--job_id', type=str, default= '')

    args = parser.parse_args()
    mention_type = args.mention_type
    job_id = args.job_id

    additional_sampling = False
    special_tokens = ["E"]

    end_start_special_tokens = []
    for token in special_tokens:
        end_start_special_tokens.append(f"[{token}]")
        end_start_special_tokens.append(f"[/{token}]")

    ## Setting parameter values
    #learning rate params
    lr_start_value = 2e-8
    lr_end_value = 2e-6
    num_values = 7 
    intermediate_values = np.logspace(np.log10(lr_start_value), np.log10(lr_end_value), num=num_values)

    #Fix test dataset
    language_model = "bart-large"
    tokenizer = create_tokenizer(language_model,end_start_special_tokens)
    # ECB+ test
    cl_cdcr_dataset = CL_Dataset(name="ecb_test",data_paths=['../CDCR/data/ECBplus-prep/output_data/test'],context_method='hsu',
        mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="inference") 
    
    while True: #Start tuning
        #Choosing hyperparameters randomly
        
        learning_rate = random.choices(intermediate_values, k=1)[0]
        margin = random.choices([0.25,0.3,0.35,0.4,0.45,0.5,0.55], k=1)[0]
        #Batch size: try 64 and 128. Initially 128
        threshold = random.choices([0.10,0.15,0.20,0.25,0.30], k=1)[0]

        #Chosen params
        df_current = pd.DataFrame({"mention_type":mention_type,"learning_rate":learning_rate,"language_model":language_model,
                                   "margin":margin,"threshold":threshold,"job_id":job_id},index=[0])
        
        #Check if these params have been or are being tested:
        #if file does not exist, create it
        file_path = "df_hyperparams.csv"
        if os.path.isfile(file_path):
            df_hyperparams = pd.read_csv(file_path)
        else:
            df_hyperparams = pd.DataFrame(columns = ["mention_type","learning_rate","language_model",
                            "margin","threshold","dev_conll","dev_lea",
                            "dev_pd_lea","test_conll","test_lea","test_pd_lea","job_id"])
            
        merged_df = pd.merge(df_hyperparams, df_current, on=["mention_type", "learning_rate", "language_model", "margin", "threshold"], how="inner")
        if merged_df.empty:
            # Values from df1 do not exist in df, create a new entry
            df_hyperparams = pd.concat([df_hyperparams, df_current], ignore_index=True)
            df_hyperparams.to_csv("df_hyperparams.csv",index=False)
            
            print("New entry added to df hyperparams")
        else:
            print("Values from df1 already exist in df. Generating new params")
            continue

        # Check if the file is not writen yet
        config = {
            #constant
            "data_setup":0,
            "context_method": 'hsu',
            "mention_type" : "event",
            "language_model_name" : language_model, #Prob distribution for language models
            "special_tokens": special_tokens,
            "additional_sampling" : additional_sampling,
            "val_batch_size":256,
            #Variables
            "learning_rate": learning_rate, 
            "margin":margin,
            "train_batch_size":128, # For now
            "clustering_threshold":threshold,
            "max_epochs":100,
            "patience":13
        }
        #train with current setup

        #send model
        
        best_model_state,best_conll,best_lea,best_pd_lea = train_hyperparam_tuning(config)

        #Predict on ECB+ test
        dataloader = create_data_loader(cl_cdcr=cl_cdcr_dataset,batch_size=256)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Would the test model not be the same as the last that was stored? I should pass it as parameter!!
        test_model = create_siamese_model(language_model,tokenizer)
        test_model.load_state_dict(best_model_state)
        test_model.to(device)

        df_metrics_test, _ = predict_and_evaluate(dataloader=dataloader,device=device,model=test_model,cl_cdcr_dataset=cl_cdcr_dataset,metrics="all",without_singletons=True)
        
        conll_test = df_metrics_test.loc["conll","F1"]
        lea_test = df_metrics_test.loc["lea","F1"]
        pd_lea_test = df_metrics_test.loc["pd_lea","F1"]
        
        df_current = pd.DataFrame({"mention_type":mention_type,"learning_rate":learning_rate,"language_model":language_model,
                                   "margin":margin,"threshold":threshold,"dev_conll":best_conll,"dev_lea":best_lea,
                                   "dev_pd_lea":best_pd_lea,"test_conll":conll_test,"test_lea":lea_test,"test_pd_lea":pd_lea_test,"job_id":job_id},index=[0])
        

        #complete data from the initial input:
        df_hyperparams = pd.read_csv("df_hyperparams.csv")
        merged_df = pd.merge(df_hyperparams, df_current, on=["mention_type", "learning_rate", "language_model", "margin", "threshold"], how="inner")
        if merged_df.empty:
            raise Exception("These parameters should already exist in the table")
        else:
            df_hyperparams = pd.concat([df_hyperparams,df_current])
            df_hyperparams = df_hyperparams.drop_duplicates(subset=["mention_type", "learning_rate", "language_model", "margin", "threshold"],keep="last")
            df_hyperparams.to_csv("df_hyperparams.csv",index=False)


