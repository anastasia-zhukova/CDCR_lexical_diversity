
from torch.autograd import Variable #This is deprecated, https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    #I modified it from the original because I won't use another sequence but a binary label
    def __init__(self, src, trg, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.trg = trg
    

### encoder
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, nhead, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(nhead, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1, return_layer=None):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout), num_layers)
        self.norm = LayerNorm(d_model)
        self.return_layer = return_layer

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        mid_encoder_feature = None
        for idx, layer in enumerate(self.layers):
            x = layer(x, mask)
            if idx + 1 == self.return_layer:
                mid_encoder_feature = self.norm(x)
        return self.norm(x), mid_encoder_feature


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, nhead, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(nhead, d_model)
        self.src_attn = MultiHeadedAttention(nhead, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout), num_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

# From BERT huggingface
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size=512, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, layer_norm_eps=1e-12):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size=hidden_size, layer_norm_eps=layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, double=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.double = double

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.double and x.size(1) % 2 != 0: #Check if it is even
            lenx_1 = int(x.size(1)/2)
            lenx_2 = lenx_1 + 1
            x = x + Variable(torch.cat((self.pe[:, :lenx_1], self.pe[:, :lenx_2]), 1), requires_grad=False)
        else:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        super(BinaryClassifier, self).__init__()
        self.prediction_head = BertPredictionHeadTransform(hidden_size, layer_norm_eps=layer_norm_eps)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,final_encoder_layer):
        transformed_states = self.prediction_head(final_encoder_layer)
        first_token_hidden_state = transformed_states[:, 0, :] #Takes special token /s
        output = self.fc(first_token_hidden_state)
        output = self.sigmoid(output)
        return output
    

class PGKPR(nn.Module):
    def __init__(self, d_model, vocab, nhead=8, num_encoder=6, num_decoder=6,
                 d_ff=2048, dropout=0.1, mid_loss=True, double_pe=False,
                 shared_emb=False):
        super(PGKPR, self).__init__()
        self.is_mid_loss = mid_loss
        self.d_model = d_model
        self.shared_emb = shared_emb
        self.src_embed_pos = nn.Sequential(Embeddings(d_model, vocab), #Token id to embedding
                                           PositionalEncoding(d_model, dropout, double=double_pe))

        #d_ff seems to be the hidden layer size of the FNN
        self.encoder = Encoder(num_layers=num_encoder, d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, return_layer=6)

        #This BertClassification layer seems to return an output per token, but the logic could be 
        #Useful to decode to a binary output
        self.decoder = BinaryClassifier(hidden_size=d_model, layer_norm_eps=1e-12)

        self.parameters_init()

    def encode(self, src, src_mask):

        embedding = self.src_embed_pos(src)
        
        return self.encoder(embedding, src_mask)

    def decode(self, final_encoder_layer):
        return self.decoder(final_encoder_layer)
    
    def forward(self, src,src_mask):
        """Take in and process masked src and target sequences."""
        #
        encode_out, _ = self.encode(src, src_mask)
        #encode_out is the final encoded layer and pre_encode_out an intermediate one defined by return layer
        x = self.decode(encode_out)
        return x



    def parameters_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)