import argparse
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, logging, RobertaModel
import torch

from dataset import CL_Dataset, CustomSubset
from evaluation import compute_cdcr_metrics
from model import SiameseNetwork, CustomContrastiveLoss

logging.set_verbosity_error()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define arguments
    parser.add_argument('--train_conll_paths', type=str, nargs='+', default=['../CDCR/data/ECBplus-prep/output_data/train'])
    parser.add_argument('--val_conll_paths', type=str, nargs='+', default=['../CDCR/data/ECBplus-prep/output_data/val'])
    parser.add_argument('--mention_type', type=str, default='event')
    parser.add_argument('--context_method', type=str, default='hsu')
    parser.add_argument('--train_dataset_name', type=str, default='ecbplus_train')
    parser.add_argument('--val_dataset_name', type=str, default='ecbplus_val')
    parser.add_argument('--language_model', type=str, default='roberta-large')
    parser.add_argument('--special_tokens', type=str, nargs='+', default=['[E]', '[/E]'])
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    #Job ID to check logs
    parser.add_argument('--job_id', type=str, default= '')
    ## Parameters
    #Data
    # Parse arguments
    args = parser.parse_args()

    train_conll_paths = args.train_conll_paths
    val_conll_paths = args.val_conll_paths
    mention_type = args.mention_type
    context_method = args.context_method
    train_dataset_name = args.train_dataset_name
    val_dataset_name = args.val_dataset_name
    language_model = args.language_model
    special_tokens = args.special_tokens
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    job_id = args.job_id
    #num_epochs = 100 for events and 50 for entities
    num_epochs = args.num_epochs


    #Model
    gpu_num = [0]
    margin=0.4

    #Optimizer and learning
    learning_rate = 2e-7
    weight_decay = 1e-2


    print("Mention type:",mention_type )
    print("Language model:",language_model )
    print("Job id:",job_id )
    print("Number of epochs",num_epochs )
    print("train batch size",train_batch_size)
    print("val_batch_size",val_batch_size )
    print("------------")

    #Initialize model and tokenizer
    roberta = RobertaModel.from_pretrained(pretrained_model_name_or_path= language_model) 
    tokenizer = RobertaTokenizer.from_pretrained(language_model)
    tokenizer.add_tokens(special_tokens)

    #Initialization
    #Load data
    # ECB+ topic 34 anomaly is accounted for in the samples
    cl_cdcr_train = CL_Dataset(name=train_dataset_name,data_paths=train_conll_paths,context_method=context_method,
                            mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=special_tokens,mode="train") 

    cl_cdcr_val = CL_Dataset(name=val_dataset_name,data_paths=val_conll_paths,context_method=context_method,
                            mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=special_tokens,mode="inference") 

    train_dataloader= DataLoader(cl_cdcr_train, batch_size=train_batch_size, shuffle=True)

    val_dataloader= DataLoader(cl_cdcr_val, batch_size=val_batch_size, shuffle=False)

    # Initialize model
    device_ids = gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    siamese_model_single = SiameseNetwork(model=roberta,tokenizer=tokenizer,device=device).to(device)
    model=siamese_model_single
    # To paralelize
    #siamese_model = torch.nn.DataParallel(siamese_model_single, device_ids=device_ids)
    #model = siamese_model_single

    optimizer_params = [
    {'params': model.language_model.parameters(), 'lr': learning_rate},
    {'params': model.mlp_cls_embed.parameters(), 'lr': learning_rate},
    {'params': model.mlp_mention_embed.parameters(), 'lr': learning_rate},
]

    contrastive_loss = CustomContrastiveLoss(margin=margin)
    optimizer = torch.optim.AdamW(optimizer_params)

    #Training loop

    best_conll_f1 = 0.0
    model_path = os.path.join(os.getcwd(), "models/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(num_epochs):
        out_dir = os.path.join(model_path,f"{mention_type}",f"{language_model}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        accumulate_loss = 0
        model.train()

        #Train dataloader
        for array_mention1,array_mention2, labels in train_dataloader:


            optimizer.zero_grad()  # Zero the gradients
            output_mentions1 = model(array_mention1)
            output_mentions2 = model(array_mention2)

            labels = labels.to(device)            
            loss = contrastive_loss(output_mentions1, output_mentions2, labels)

            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Optimization step

            torch.cuda.empty_cache()
            accumulate_loss += loss.item()
        
        print(f'epoch {epoch+1}/{num_epochs}, Train accumulate loss: {accumulate_loss}')

        #Validation
        model.eval()
        #Store vector representations by topic
        df_metrics, _ = compute_cdcr_metrics(cl_cdcr_dataset=cl_cdcr_val,dataloader=val_dataloader,model=model,metrics="all")
        # Store the best model
        conll_f1 = df_metrics.loc["conll","F1"]
        print(f'epoch {epoch+1}/{num_epochs}, eval conll: {conll_f1}')
        if conll_f1 > best_conll_f1:
            best_conll_f1 = conll_f1
            torch.save(model.state_dict(), out_dir+f"/siamese_model_batch_{train_batch_size}_epochs_{num_epochs}.pth")
            print(f'epoch {epoch+1}/{num_epochs}, model saved')

