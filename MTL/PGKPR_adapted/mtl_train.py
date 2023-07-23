from tqdm import tqdm
from architecture import Embeddings, PositionalEncoding, Encoder, Batch,BinaryClassifier, PGKPR
import torch.nn as nn
import torch
import math
from utils import loadVocab, vocab_path

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#Load data
train_inputs,val_inputs, test_inputs = torch.load("train_inputs.pt"),torch.load("val_inputs.pt"),torch.load("test_inputs.pt")
train_labels,val_labels, test_labels = torch.load("train_labels.pt"),torch.load("val_labels.pt"),torch.load("test_labels.pt")

#Load tokenizer
LOWERCASE = True
token2idx, idx2token = loadVocab(vocab_path, lowercase=LOWERCASE) 
vocab_size = len(token2idx)

#Define model parameters
D_MODEL = 512

MID_LOSS = True
DOUBLE_POSITION = False
tqdm_print_step = 1
model_save_path = 'models/'
BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

# Load model and plug optimizer
model = PGKPR(d_model=D_MODEL, vocab=vocab_size, mid_loss=MID_LOSS, double_pe=DOUBLE_POSITION,
                          shared_emb=True)


model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

#Pack to tuples 
train_data = train_inputs, train_labels 
val_data = val_inputs, val_labels
test_data = test_inputs, test_labels

train_data = TensorDataset(*train_data)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_data = TensorDataset(*val_data)
test_data = TensorDataset(*test_data)

val_dataloader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=VAL_BATCH_SIZE)  

dataloader = {'train': train_dataloader, 'val': val_dataloader, 'test':test_dataloader}


######################
#Run training
def run_once(model, batch,loss_function,optimizer,is_training):
    predicted_probs= model.forward(src=batch.src,src_mask=batch.src_mask)
    predicted_probs = predicted_probs.squeeze()

    loss = loss_function(predicted_probs, batch.trg)
    
    if is_training and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

loss_function = torch.nn.BCELoss()
best_loss = float("inf")

for epoch in range(5): #TODO define epochs parameter from config file
    # scheduler.step()
    print(f'\n EPOCH {epoch}')
    model.train()
    total_loss = 0
    #cdcr_loss, pi_loss= 0, 0
    
    for phase in ['train', 'val']:
        if phase=='train':
            model.train()
            is_training= True
        else:
            model.eval()
            is_training= False

        with tqdm(total=len(dataloader[phase])) as pbar: 
            #Dataloader should be controlled to send a batch of one task only.
            for i, data in enumerate(dataloader[phase]):
                src, tgt= data #dataloader must have source and target, in my case the label
                batch = Batch(src, tgt)
                # src, tgt= src.cuda() tgt.cuda()#TODO send everything to device. I could adapt it to finding device first
                binary_loss = run_once(model,batch,loss_function,model_opt,is_training)
                total_loss += binary_loss #It would be a good idea to monitor both losses separately
                
                if i % tqdm_print_step == 0:
                        pbar.update(tqdm_print_step)
                        pbar.set_description(
                            f'train loss: {binary_loss:.4f}')
                        
        epoch_loss = total_loss / len(dataloader[phase]) #Mean relative to dataloader size

        print(f"epoch loss: {epoch_loss}")

        if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
            print("save model")
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{model_save_path}/epoch{epoch}_loss_{epoch_loss}.pkl')