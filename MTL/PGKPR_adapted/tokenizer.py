from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

#Load encoder
tokenizer = Tokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

#post processing template
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A", #Closing [SEP] is added in the CDCR preprocessing
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
#Padding
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]",length=350) #In ECB+ there was a tokenized sequence of length 315.

# Define the list of special tokens to add
special_tokens = ["[START]", "[END]"]

# Add the special tokens to the tokenizer
tokenizer.add_tokens(special_tokens)