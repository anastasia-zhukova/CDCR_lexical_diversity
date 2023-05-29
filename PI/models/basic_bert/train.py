#import libraries
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer
import torch
from utils import *
import os


if __name__ == "__main__":
    print("Start training")
    
    ### LEARNING PARAMETERS ###
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU available?
    print(f"Training with: {device}")
    
    saved_models_path = "./saved_models"
    if not os.path.exists(saved_models_path):
        os.makedirs(saved_models_path)
        print(f"Created path for models in {saved_models_path}")
    

    ### FIX SEEDS ###
    
    #fix_random_seed(26, device=device)

    ### DATA ####
    
    dataset = load_dataset('glue', 'mrpc')
    # tokenize dataset
    tokenized_dataset = reformat_dataset(dataset)

    ### MODEL ####
    
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

    ### TRAINING LOOP ###
    # Define the Trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)
    trainer.train()
    print("Finished training.")
    
    #Save model state binary
    torch.save(model.state_dict(), 'state.pt')
    torch.save(model, 'model.pt')
