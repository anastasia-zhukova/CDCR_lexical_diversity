import argparse
import os
from transformers import logging
import torch

from dataloader import create_data_loader
from dataset import CL_Dataset
from evaluation import predict_and_evaluate
from model import CustomContrastiveLoss, create_tokenizer, create_siamese_model, create_optimizer

logging.set_verbosity_error()
#TODO set fixed setups of training and val paths instead of passing the paths by parameter
#Then, add that setup identifier to the model's path
#Functions
def train(gpu_id,args):
    #Initialize backend
    world_size=args.world_size

    #Initialization
    language_model_name = args.language_model
    num_epochs = args.num_epochs #num_epochs = 100 for events and 50 for entities
    print("Number of epochs",num_epochs )

    #Load special tokens
    special_tokens = args.special_tokens
    end_start_special_tokens = []
    for token in special_tokens:
        end_start_special_tokens.append(f"[{token}]")
        end_start_special_tokens.append(f"[/{token}]")

    #Create model path
    out_dir = os.path.join(os.getcwd(),"models",args.mention_type,language_model_name,f"tokens_{'_'.join(special_tokens)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Create tokenizer
    tokenizer = create_tokenizer(language_model_name,end_start_special_tokens)

    #create dataloaders
    context_method = args.context_method
    mention_type = args.mention_type
    export_samples = args.export_samples

    cl_cdcr_train = CL_Dataset(name=args.train_dataset_name,data_paths=args.train_conll_paths,context_method=context_method,
                        mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="train",export_samples=export_samples) 

    cl_cdcr_val = CL_Dataset(name=args.val_dataset_name,data_paths=args.val_conll_paths,context_method=context_method,
                    mention_type=mention_type,tokenizer=tokenizer, special_mention_tokens=end_start_special_tokens,mode="inference") 
 
    train_dataloader = create_data_loader(cl_cdcr=cl_cdcr_train,batch_size=args.train_batch_size)

    val_dataloader = create_data_loader(cl_cdcr=cl_cdcr_val,batch_size=args.val_batch_size)

    #Create model
    model = create_siamese_model(language_model_name,tokenizer)
    device = torch.device('cuda:0')
    model.to(device)

    #Optimizer and loss
    optimizer = create_optimizer(args.learning_rate,model)
    contrastive_loss = CustomContrastiveLoss(margin=args.margin)
    
    #Distributed training
    if world_size>1:
        os.environ['RANK'] = str(gpu_id)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size) #nccl is the backend for GPU only
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[i for i in range(world_size)])

    

    #Training loop
    best_conll_f1 = 0.0
    for epoch in range(num_epochs):
        
        accumulated_loss = 0
        model.train()
        for input_ids1, attention_masks1,input_ids2,attention_masks2, labels in train_dataloader:
            optimizer.zero_grad()  # Zero the gradients
            input_ids1, attention_masks1,input_ids2,attention_masks2 = input_ids1.to(device), attention_masks1.to(device),input_ids2.to(device),attention_masks2.to(device)
            labels = labels.to(device)    

            output_mentions1 = model(input_ids1, attention_masks1)
            output_mentions2 = model(input_ids2, attention_masks2)

            loss = contrastive_loss(output_mentions1, output_mentions2, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.empty_cache()
            accumulated_loss += loss.item()

        print(f'epoch {epoch+1}/{num_epochs}, Train accumulated loss: {accumulated_loss}')

        # #Validation
        # val_model = None
        # if world_size>1:
        #     #Start model from scratch
        #     # Create a model_state_dict without the 'module.' prefix added by parallel module
        #     model_state_dict = {k.replace('module.', ''): v for k, v in model.state_dict().items()}
        #     new_model = create_siamese_model(language_model_name,tokenizer)
        #     new_model.load_state_dict(model_state_dict)

        #     new_model.to(device)
        #     val_model = new_model
        # else:
        #     val_model = model
        
        # val_model.eval()
        #df_metrics, _ = predict_and_evaluate(dataloader=val_dataloader,device=device,model=val_model,cl_cdcr_dataset=cl_cdcr_val,metrics="all")
        model.eval()
        df_metrics, _ = predict_and_evaluate(dataloader=val_dataloader,device=device,model=model,cl_cdcr_dataset=cl_cdcr_val,metrics="all")
        #Print epoch results
        val_conll_f1 = df_metrics.loc["conll","F1"]
        print(f'epoch {epoch+1}/{num_epochs}, eval conll: {val_conll_f1}')

        # Store the best model
        if val_conll_f1>best_conll_f1:
            best_conll_f1=val_conll_f1
            torch.save(model.state_dict(), os.path.join(out_dir, f"siamese_model_batch_{args.train_batch_size}_epochs_{num_epochs}.pth"))

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
    parser.add_argument('--special_tokens', type=str, nargs='+', default=["E"])
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-7, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.4, help='margin')
    parser.add_argument('--export_samples', action='store_true', default=False, help='Export samples or not')
    #Job ID to check logs
    parser.add_argument('--job_id', type=str, default= '')
    # Number of GPUs available
    parser.add_argument('--world_size', default=1, type=int, 
                        help='number of gpus for distributed training')
    
    # Parse arguments
    args = parser.parse_args()
    world_size = args.world_size
    
    if world_size>1:

        #Parallel processing
        ngpus_per_node = torch.cuda.device_count()
        print(f"there are {ngpus_per_node} GPUs available")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501" 
        torch.multiprocessing.spawn(fn=train,args=(args,), nprocs = 2)
    else:
        train(0,args)