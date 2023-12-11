
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AdamW,RobertaModel,RobertaTokenizer
from torch.utils.data import DataLoader,TensorDataset,RandomSampler

# function split train dataset into train, validation and test sets
def train_test (text,labels,test_size):
  train_text, test_text, train_labels, test_labels = train_test_split(text, 
                                                                    labels, 
                                                                    random_state=2018, 
                                                                    test_size=test_size,
                                                                    stratify=labels)
  return train_text, test_text, train_labels, test_labels

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(sentences,labels,max_length = None):
  "tokenizes input and returns tokenized input + labels as tensors"

  input_ids = []
  attention_masks = []

  for text in sentences.to_list():
      encodings = tokenizer.encode_plus(text,add_special_tokens = True,max_length = max_length
                                        ,truncation = True, padding = 'max_length',return_attention_mask = True)
      input_ids.append(encodings['input_ids'])
      attention_masks.append(encodings['attention_mask'])

  return torch.tensor(input_ids),torch.tensor(attention_masks),torch.tensor(labels.to_list())


# function to get predictions for test data
def predict(model,dataloader):

  predictions = []
  for batch in dataloader:
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch
    with torch.no_grad():
      output = model(sent_id, attention_mask=mask,labels = labels)
      preds = output[1]
      preds = preds.detach().cpu().numpy()
      predictions.append(np.argmax(preds, axis = 1).flatten())

  #merge sublists of predictions
  predictions = [label for batch in predictions for label in batch]

  return predictions

#set seed
np.random.seed(0)
torch.manual_seed(0)   
random.seed(0)    
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#read WNC corpus 
df_bias = pd.read_excel('dapt_bias_dataset.xlsx')
df_bias.dropna(inplace=True)

#train test split + tokenization
train_text, test_text, train_labels, test_labels = train_test(df_bias['text'], df_bias['label_bias'],0.2)
train_input_ids,train_attention_masks,train_y = tokenize(train_text, train_labels)
test_input_ids,test_attention_masks,test_y = tokenize(test_text,test_labels) 
train_data_wiki = TensorDataset(train_input_ids, train_attention_masks, train_y)
test_data_wiki = TensorDataset(test_input_ids, test_attention_masks, test_y)

epochs = 1
batch_size = 32

train_sampler = RandomSampler(train_data_wiki)
test_sampler = RandomSampler(test_data_wiki)

train_dataloader = DataLoader(train_data_wiki,sampler= train_sampler, batch_size=batch_size)
test_dataloader = DataLoader(test_data_wiki,sampler= test_sampler, batch_size=batch_size)

#define loss
cross_entropy = nn.CrossEntropyLoss()

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.vocab_transform = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier1 = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, labels):
        output_1 = self.roberta(input_ids, attention_mask)['last_hidden_state']
        pooler = output_1[:, 0, :]
        pooler = self.vocab_transform(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier1(pooler)
        loss = cross_entropy(output, labels)

        return loss
    

#connect to GPU
if torch.cuda.is_available():       
    device = torch.device("cuda:0")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = RobertaClass()

model = model.to(device)
optim = AdamW(model.parameters(), lr=1e-5)


#train function
def train(dataloader):

  model.train()
  total_loss = 0
  counter = 0
    
  for _,batch in enumerate(dataloader):
    counter += 1
    sys.stdout.write('\r Batch {}/{}'.format(counter,len(dataloader)))
    print('\r Batch {}/{}'.format(counter,len(dataloader)))
    optim.zero_grad()
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch
    loss = model(sent_id, attention_mask=mask,labels = labels)
    loss.backward() 
    total_loss = total_loss+loss.item()
    optim.step()
    del batch,sent_id,mask,labels
        
  avg_loss = total_loss / len(dataloader)
  return avg_loss


#test function

def validate(dataloader):
    model.eval()
    total_loss = 0
    print("\nValidating...")
    counter = 0
    for batch in dataloader:
      counter +=1
      batch = [r.to(device) for r in batch]
      sent_id, mask, labels = batch

      with torch.no_grad():
        loss = model(sent_id, attention_mask=mask,labels = labels)
        total_loss = total_loss+loss

    avg_loss = total_loss / len(dataloader) 
    return avg_loss


#train/validate function

def train_validate(train_dataloader,test_dataloader):
  best_valid_loss = float('inf')

  # empty lists to store training and validation loss of each epoch
  train_losses=[]
  valid_losses=[]

  #for each epoch
  for epoch in range(epochs):
        
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss = train(train_dataloader)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    
    #evaluate model
    valid_loss = validate(test_dataloader)
    
    #save the best model
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'pytorch_model.bin') #insert path here
      
    #if validation loss increases, stop training
    elif valid_loss >= best_valid_loss:
      print("\n Validation loss not decreased, Model of previous epoch saved")
      break
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


#apply training and validation
train_validate(train_dataloader,test_dataloader)