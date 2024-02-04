import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaTokenizer, RobertaModel

class SiameseNetwork(nn.Module):    

    def __init__(self, model,tokenizer, mlp_hidden_dim=1024):        
        
        super(SiameseNetwork, self).__init__()

        self.tokenizer = tokenizer
        self.language_model = model  #Initialized model
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.mlp_hidden_dim = mlp_hidden_dim

        # MLP layers
        self.mlp_cls_embed = self.build_mlp()
        self.mlp_mention_embed = self.build_mlp()

    def build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim)
        )
    
    def forward(self, batch_input_ids,batch_attention_mask):        
        # Encode sentences        
        cls_embed, mention_embed = self.encode_sentence(batch_input_ids,batch_attention_mask)

        # Pass through MLP layers        
        output = self.mlp_concat(cls_embed,mention_embed)
        return output
    
    def encode_sentence(self, batch_input_ids,batch_attention_mask):

        # Extract indices of [E] and [/E] tokens
        e_start_idx = batch_input_ids.eq(self.tokenizer.convert_tokens_to_ids('[E]')).nonzero(as_tuple=True)
        e_end_idx = batch_input_ids.eq(self.tokenizer.convert_tokens_to_ids('[/E]')).nonzero(as_tuple=True)
        

        #Check that there special mentions only appear once per sequence
        # Validate if there are no repeated values
        mention_tokens_appear_once = len(e_start_idx[0]) == len(torch.unique(e_start_idx[0])) and len(e_end_idx[0]) == len(torch.unique(e_end_idx[0])) 
        mention_tokens_exist = len(e_start_idx[0]) == len(e_end_idx[0]) and len(e_start_idx[0])==batch_input_ids.shape[0]
        
        if not mention_tokens_appear_once or not mention_tokens_exist:
            raise Exception("Mention tokens [E] [/E] are not unique in each sequence from the batch")
        outputs = self.language_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        # Get the output embeddings from the last layer
        last_hidden_states = outputs.last_hidden_state


        # Mask tokens outside the mention

        indices = torch.arange(last_hidden_states.size(1),device=last_hidden_states.device) #Create indices for each token (dimension 1)
        
        # Create masks based on start and end indices
        start_mask = indices >= e_start_idx[1].unsqueeze(1) # dim 1 is # tokens
        end_mask = indices <= e_end_idx[1].unsqueeze(1)

        # Combine masks to get the final mask
        mask = start_mask & end_mask
        # Element-wise multiplication with the mask
        masked_hidden_states = last_hidden_states * mask.unsqueeze(2).float() #unsqueeze to match third dimension
        #Element wise sum of mention token embeddings
        summed_mention_embeddings = masked_hidden_states.sum(dim=1)
        
        cls_token_embeddings = last_hidden_states[:, 0, :]

        return cls_token_embeddings, summed_mention_embeddings

    def mlp_concat(self,cls_embed,mention_embed):

        output_cls = self.mlp_cls_embed(cls_embed)
        output_mention = self.mlp_mention_embed(mention_embed)
        concat_output = torch.cat([output_cls,output_mention],dim=1)
        return concat_output
    

class CustomContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomContrastiveLoss, self).__init__()
        self.margin = margin #0.4 for events, 0.7 for entities

    def forward(self, u, v, label):
        # Compute distance
        distance = 1 - F.cosine_similarity(u, v)

        # Compute contrastive loss
        loss = label.float()  * distance.pow(2) + (1 - label).float() * F.relu(self.margin - distance).pow(2)

        return loss.mean()  # Take the mean over the batch


def create_tokenizer(language_model_name,special_tokens):
    if language_model_name=="da-roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("da-roberta tokenizer loaded")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(language_model_name)
    tokenizer.add_tokens(special_tokens)
    return tokenizer


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.vocab_transform = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier1 = torch.nn.Linear(768,2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.vocab_transform(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier1(pooler)

        return output
    

def create_siamese_model(language_model_name,tokenizer_instance):
    if language_model_name=="da-roberta":
        weight_dict = torch.load('DAPT_models/DA-Roberta.bin')
        model = RobertaClass()
        model.load_state_dict(weight_dict)
        language_model_instance = model.roberta
        print("loaded da-roberta model")
    else:
        language_model_instance = RobertaModel.from_pretrained(pretrained_model_name_or_path= language_model_name) 

    siamese_model = SiameseNetwork(model=language_model_instance,tokenizer=tokenizer_instance)
    return siamese_model

def create_optimizer(learning_rate,model):
    # Unwrap DataParallel if it's used
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    #Create optimizer
    optimizer_params = [
    {'params': model.language_model.parameters(), 'lr': learning_rate},
    {'params': model.mlp_cls_embed.parameters(), 'lr': learning_rate},
    {'params': model.mlp_mention_embed.parameters(), 'lr': learning_rate},
]
    optimizer = torch.optim.AdamW(optimizer_params)
    return optimizer