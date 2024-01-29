import argparse
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, logging, RobertaModel
import torch

from dataset import CL_Dataset, CustomSubset
from evaluation import compute_cdcr_metrics
from model import SiameseNetwork, CustomContrastiveLoss

logging.set_verbosity_error()

#Functions

def init_language_model_tokenizer(language_model_name,special_tokens):
    language_model = RobertaModel.from_pretrained(pretrained_model_name_or_path= language_model_name) 
    tokenizer = RobertaTokenizer.from_pretrained(language_model)
    tokenizer.add_tokens(special_tokens)
    return language_model, tokenizer

def create_model(language_model_instance,tokenizer):
    siamese_model = SiameseNetwork(model=language_model_instance,tokenizer=tokenizer)
    return siamese_model
    
def create_data_loaders(args,tokenizer):
    
    context_method = args.context_method
    special_tokens = args.special_tokens
    export_samples = args.export_samples
    mention_type = args.mention_type

    cl_cdcr_train = CL_Dataset(name=args.train_dataset_name,data_paths=args.train_conll_paths,context_method=context_method,
                        mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=special_tokens,mode="train",export_samples=export_samples) 

    cl_cdcr_val = CL_Dataset(name=args.val_dataset_name,data_paths=args.val_conll_paths,context_method=context_method,
                            mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=special_tokens,mode="inference",export_samples=export_samples) 

    train_dataloader= DataLoader(cl_cdcr_train, batch_size=args.train_batch_size, shuffle=True)

    val_dataloader= DataLoader(cl_cdcr_val, batch_size=args.val_batch_size, shuffle=False)

    return train_dataloader, val_dataloader, cl_cdcr_val

def validate(val_dataset,val_dataloader,language_model_instance, tokenizer, model_state_dict,gpu_id):
    #Start model from scratch
    device = torch.device(f'cuda:{gpu_id}')
    model = create_model(language_model_instance,tokenizer)
    model.load_state_dict(model_state_dict)
    model.to(device)

    model.eval()

    #Store vector representations by topic
    df_metrics, _ = compute_cdcr_metrics(cl_cdcr_dataset=val_dataset,dataloader=val_dataloader,model=model,metrics="all")
    # Store the best model
    conll_f1 = df_metrics.loc["conll","F1"]

    return conll_f1

def create_optimizar_and_loss(args,model):
    #Create optimizer
    learning_rate = args.learning_rate
    optimizer_params = [
    {'params': model.language_model.parameters(), 'lr': learning_rate},
    {'params': model.mlp_cls_embed.parameters(), 'lr': learning_rate},
    {'params': model.mlp_mention_embed.parameters(), 'lr': learning_rate},
]
    optimizer = torch.optim.AdamW(optimizer_params)

    #Create loss
    margin=args.margin
    contrastive_loss = CustomContrastiveLoss(margin=margin)

    return optimizer, contrastive_loss
    

def optimizer_step(model,optimizer,train_dataloader,loss_fn,device):

    accumulate_loss = 0
    for array_mention1,array_mention2, labels in train_dataloader:

        optimizer.zero_grad()  # Zero the gradients

        transposed_array_mention1 = array_mention1.transpose(0, 1).to(device)
        transposed_array_mention2 = array_mention2.transpose(0, 1).to(device)

        output_mentions1 = model(transposed_array_mention1)
        output_mentions2 = model(transposed_array_mention2)

        labels = labels.to(device)            
        loss = loss_fn(output_mentions1, output_mentions2, labels)

        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # Optimization step

        torch.cuda.empty_cache()
        accumulate_loss += loss.item()
    
    return accumulate_loss

def train(gpu_id,args):
    #Initialize backend
    torch.distributed.init_process_group(backend='nccl', init_method='env://') #nccl is the backend for GPU only
    
    #Initialization
    language_model_name = args.language_model
    num_epochs = args.num_epochs #num_epochs = 100 for events and 50 for entities
    print("Number of epochs",num_epochs )

    out_dir = os.path.join(os.getcwd(),"models/",f"{args.mention_type}",f"{language_model_name}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Create model
    world_size = 2 #Probably don't need it, I could use NumberByParameter
    language_model_instance, tokenizer = init_language_model_tokenizer(language_model_name,args.special_tokens)
    device = torch.device('cuda:0')
    model = create_model(language_model_instance,tokenizer)
    model = torch.nn.parallel.DataParallel(model, device_ids=[i for i in range(world_size)])
    model.to(device)

    model.train()
    #Optimizer and loss
    optimizer, contrastive_loss = create_optimizar_and_loss(args,model)

    #Training loop
    best_conll_f1 = 0.0
    for epoch in range(num_epochs):
        #create dataloaders
        train_dataloader, val_dataloader, cl_cdcr_val = create_data_loaders(args,tokenizer)
        accumulate_loss = optimizer_step(model=model,optimizer=optimizer,train_dataloader=train_dataloader,loss_fn=contrastive_loss,device=device)

        # Create a model_state_dict without the 'module.' prefix added by parallel module
        model_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}

        val_conll_f1 = validate(val_dataset=cl_cdcr_val,val_dataloader=val_dataloader,language_model_instance=language_model_instance,
                                tokenizer=tokenizer,model_state_dict=model_state_dict,gpu_id=gpu_id)

        #Print epoch results
        print(f'epoch {epoch+1}/{num_epochs}, Train accumulate loss: {accumulate_loss}')
        print(f'epoch {epoch+1}/{num_epochs}, eval conll: {val_conll_f1}')

        if val_conll_f1>best_conll_f1:
            best_conll_f1=val_conll_f1
            torch.save(model.state_dict(), out_dir+f"/siamese_model_batch_{args.train_batch_size}_epochs_{num_epochs}.pth")
            print(f'epoch {epoch+1}/{num_epochs}, model saved')


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
    parser.add_argument('--learning_rate', type=float, default=2e-7, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='margin')
    parser.add_argument('--export_samples', action='store_true', default=False, help='Export samples or not')
    #Job ID to check logs
    parser.add_argument('--job_id', type=str, default= '')
    
    # Parse arguments
    args = parser.parse_args()

    job_id = args.job_id
    print("Job id:",job_id )

    torch.multiprocessing.spawn(fn=train, nprocs = 2)