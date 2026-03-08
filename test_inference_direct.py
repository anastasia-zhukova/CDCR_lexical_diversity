"""
Direct test of inference without Kedro logging.
"""
import sys
import pandas as pd
import torch
from pathlib import Path

# Add CL_COREF to path
CL_COREF_PATH = Path("CL_COREF")
sys.path.insert(0, str(CL_COREF_PATH))

from model import create_tokenizer, create_siamese_model

# Load mentions
print("Loading test data...")
mentions_df = pd.read_csv("cl-coref/data/03_primary/mentions_for_pairs.csv", sep=';', low_memory=False)
repr_df = pd.read_parquet("cl-coref/data/03_primary/mention_representations.parquet")

# Filter to ECBplus test split
ecb_test = mentions_df[
    (mentions_df['dataset_name'] == 'ECBplus') &
    (mentions_df['split'] == 'test')
].copy()

# Join with representations
ecb_test = ecb_test.merge(repr_df, on='mention_id', how='inner')

print(f"ECBplus test mentions: {len(ecb_test)}")

# Get one subtopic
subtopic = ecb_test['subtopic_id'].iloc[0]
subtopic_mentions = ecb_test[ecb_test['subtopic_id'] == subtopic]

print(f"\nTesting with subtopic: {subtopic}")
print(f"Mentions in subtopic: {len(subtopic_mentions)}")

# Load checkpoint
checkpoint_path = "cl-coref/data/06_models/ECBplus/siamese_model_ECBplus_epoch=06_val_f1_conll=44.23.ckpt"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['hyper_parameters'].get('config', {})

print(f"Config: {config}")

# Create model
print("\nCreating model...")
tokenizer = create_tokenizer('roberta-large', special_tokens=["[E]", "[/E]"])
model = create_siamese_model('roberta-large', tokenizer)

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"[E] token ID: {tokenizer.convert_tokens_to_ids('[E]')}")
print(f"[/E] token ID: {tokenizer.convert_tokens_to_ids('[/E]')}")

# Test encoding
print("\nTesting encoding...")
doc_contexts = subtopic_mentions['document_context'].tolist()[:2]
mention_contexts = subtopic_mentions['mention_context'].tolist()[:2]

print(f"\nSample document_context: {doc_contexts[0][:100]}...")
print(f"Sample mention_context: {mention_contexts[0][:100]}...")

# Tokenize
inputs = tokenizer(
    doc_contexts,
    mention_contexts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

print(f"\nTokenized shape: {inputs['input_ids'].shape}")
print(f"Token IDs sample: {inputs['input_ids'][0][:20]}")

# Try to encode
print("\nEncoding with model...")
try:
    with torch.no_grad():
        embeddings = model(inputs['input_ids'], inputs['attention_mask'])
        print(f"Embeddings shape: {embeddings.shape}")
        print("✓ Encoding successful!")
except Exception as e:
    print(f"✗ Encoding failed: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
