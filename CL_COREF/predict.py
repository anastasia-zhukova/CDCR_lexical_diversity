#Load libraries
import argparse
import os
import torch

from dataloader import create_data_loader
from dataset import CL_Dataset
from evaluation import predict_and_evaluate
from model import create_tokenizer, create_siamese_model

#Load model and tokenizer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    
    #Initialize model
    model_path = args.model_path
    model_path_split = model_path.split("/")

    mention_type=model_path_split[1] 
    language_model_name = model_path_split[2]
    special_tokens = model_path_split[3].split("_")[1:]
    
    end_start_special_tokens = []
    for token in special_tokens:
        end_start_special_tokens.append(f"[{token}]")
        end_start_special_tokens.append(f"[/{token}]")

    #Create and initialize model
    tokenizer = create_tokenizer(language_model_name,end_start_special_tokens)
    device = torch.device('cuda:0')
    model = create_siamese_model(language_model_name,tokenizer)
    model.load_state_dict( torch.load(model_path, map_location=device))
    model.to(device)

    #create dataloaders
    test_batch_size=128

    dict_test_data = {"ecbplus_test":['../CDCR/data/ECBplus-prep/output_data/test'], 
    "newswcl50":['../CDCR/data/NewsWCL50-prep/output_data/train',
     '../CDCR/data/NewsWCL50-prep/output_data/val',
     '../CDCR/data/NewsWCL50-prep/output_data/test']}

    #Predict
    predictions_path = os.path.join(os.getcwd(), "predictions")
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    model.eval()

    for dataset_name,data_paths in dict_test_data.items():
        cl_cdcr_dataset = CL_Dataset(name=dataset_name,data_paths=data_paths,context_method='hsu',
                mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="inference") 
        dataloader = create_data_loader(cl_cdcr=cl_cdcr_dataset,batch_size=test_batch_size)
        df_metrics, df_prediction = predict_and_evaluate(dataloader=dataloader,device=device,model=model,cl_cdcr_dataset=cl_cdcr_dataset,metrics="all")
        print(f"{dataset_name} results",df_metrics)
        print("--------------------")
        print(f"{dataset_name} test results: CoNLL F1 {round(df_metrics.loc['conll','F1'],2)} LEA F1: {round(df_metrics.loc['lea','F1'],2)}")

        file_path = os.path.join(predictions_path, f"{dataset_name}_{mention_type}.csv")
        df_prediction.to_csv(file_path)
        print(f"{dataset_name} predictions exported to {file_path}")
        
    print("---------------")
    print(f"Language model: {language_model_name}")

