#Load libraries
import argparse
import os
import torch
from torch.utils.data import DataLoader 
from transformers import RobertaTokenizer, RobertaModel

from dataset import CL_Dataset
from evaluation import compute_cdcr_metrics
from model import SiameseNetwork

#Load model and tokenizer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument('--model_path', type=str, nargs='+', default=['models/entity/roberta-large/siamese_model.pth'])
    args = parser.parse_args()

    model_path = args.model_path # By Param
    model_path_split = model_path.split("/")

    language_model = model_path_split[2] # I should read it from the model's path
    mention_type=model_path_split[1] #Read from model's path

    torch_device =  "cuda" if torch.cuda.is_available() else "cpu"
    print(torch_device)
    device = torch.device(torch_device)
    base_language_model = RobertaModel.from_pretrained(pretrained_model_name_or_path= language_model)
    # Initialize tokenizer
    special_tokens = ['[E]', '[/E]']
    tokenizer = RobertaTokenizer.from_pretrained(language_model)
    tokenizer.add_tokens(special_tokens)
    #Initialize siamese model and load weights
    model = SiameseNetwork(model=base_language_model,tokenizer=tokenizer,device=device).to(device)
    model.load_state_dict( torch.load(model_path, map_location=device))


    test_batch_size=128

    #Load data
    # Data
    ecbplus_dataset_name = "ecbplus_test"
    ecbplus_test = CL_Dataset(name=ecbplus_dataset_name,data_paths=['../CDCR/data/ECBplus-prep/output_data/test'],context_method='hsu',
                        mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=special_tokens,mode="inference") 

    ecbplus_test_dataloader= DataLoader(ecbplus_test, batch_size=test_batch_size, shuffle=False)

    newswcl50_dataset_name = "newswcl50"
    newswcl50 = CL_Dataset(name=newswcl50_dataset_name,data_paths=['../CDCR/data/NewsWCL50-prep/output_data/train',
                                                                    '../CDCR/data/NewsWCL50-prep/output_data/val',
                                                                    '../CDCR/data/NewsWCL50-prep/output_data/test'],
                                                                    context_method='hsu',mention_type=mention_type,tokenizer=tokenizer, 
                                                                    special_mention_tokens=special_tokens,mode="inference") 

    newswcl50_dataloader= DataLoader(newswcl50, batch_size=test_batch_size, shuffle=False)

    #Predict
    model.eval()

    ecbplus_test_metrics, ecbplus_test_prediction = compute_cdcr_metrics(cl_cdcr_dataset=ecbplus_test,dataloader=ecbplus_test_dataloader,model=model,metrics="all")
    newswcl50_metrics, newswcl50_prediction = compute_cdcr_metrics(cl_cdcr_dataset=newswcl50,dataloader=newswcl50_dataloader,model=model,metrics="all")

    print("ECB+ results",ecbplus_test_metrics)
    print("NewsWCL50 results",newswcl50_metrics)

    predictions_path = os.path.join(os.getcwd(), "predictions/")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)


    ecbplus_test_prediction.to_csv(f"{predictions_path}{ecbplus_dataset_name}_,{mention_type}.csv")
    newswcl50_prediction.to_csv(f"{predictions_path}{newswcl50_dataset_name}_,{mention_type}.csv")
    print("predictions exported to", predictions_path)

