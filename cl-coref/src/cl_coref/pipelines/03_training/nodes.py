"""
Nodes for training pipeline.

This pipeline trains a siamese network for coreference resolution using:
1. Filtered mention pairs from pairs_filtered_by_similarity_quantiles
2. Mention representations (embeddings/contexts) from mention_representations
3. PyTorch Lightning for training
"""

import logging
import os
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import wandb

# Add CL_COREF to path to import existing modules
CL_COREF_PATH = Path(__file__).parents[5] / "CL_COREF"
sys.path.insert(0, str(CL_COREF_PATH))

from model import create_tokenizer, create_siamese_model, CustomContrastiveLoss, create_optimizer

# Import our custom modules
from .lightning_module import CorefLightningModule
from .datamodule import CorefDataModule

logger = logging.getLogger(__name__)


def custom_tokenize_with_special_tokens(
    tokenizer,
    doc_context: str,
    mention_context: str,
    max_length: int = 128
) -> dict:
    """
    Custom tokenization that preserves [E] and [/E] markers even when truncating.

    Strategy:
    1. Tokenize mention_context separately to find [E] and [/E] positions
    2. If mention_context is too long, truncate it but preserve [E] markers
    3. Allocate remaining space to doc_context
    4. Combine sequences manually

    Args:
        tokenizer: HuggingFace tokenizer
        doc_context: Document context text (first sequence)
        mention_context: Mention context with [E] markers (second sequence)
        max_length: Maximum total sequence length

    Returns:
        Dict with 'input_ids' and 'attention_mask'
    """
    # Account for special tokens: <s> text </s> </s> text_pair </s>
    # RoBERTa uses: <s> (1 token) + text + </s> (1 token) + </s> (1 token) + text_pair + </s> (1 token)
    # Total special tokens: 4
    special_tokens_count = 4
    max_tokens_for_text = max_length - special_tokens_count

    # Tokenize mention_context to find [E] markers
    mention_tokens = tokenizer.encode(mention_context, add_special_tokens=False)

    # Find [E] and [/E] token IDs
    e_start_id = tokenizer.convert_tokens_to_ids('[E]')
    e_end_id = tokenizer.convert_tokens_to_ids('[/E]')

    # Find positions of [E] markers in mention_context
    try:
        e_start_pos = mention_tokens.index(e_start_id)
        e_end_pos = mention_tokens.index(e_end_id)
    except ValueError:
        # If markers not found, fall back to regular truncation
        return tokenizer(
            text=doc_context,
            text_pair=mention_context,
            padding='max_length',
            max_length=max_length,
            truncation='longest_first',
            return_tensors=None
        )

    # Calculate how much space mention_context needs
    mention_len = len(mention_tokens)

    # If mention_context fits, allocate rest to doc_context
    if mention_len <= max_tokens_for_text:
        doc_max_len = max_tokens_for_text - mention_len
        doc_tokens = tokenizer.encode(doc_context, add_special_tokens=False)

        # Truncate doc_context if needed
        if len(doc_tokens) > doc_max_len:
            doc_tokens = doc_tokens[:doc_max_len]
    else:
        # Mention context is too long - truncate it while preserving [E] markers
        # Strategy: Keep tokens around [E] markers, truncate the rest

        # Calculate how many tokens we can keep from mention_context
        # Reserve at least 20 tokens for doc_context (arbitrary but reasonable)
        min_doc_tokens = min(20, max_tokens_for_text // 4)
        mention_max_len = max_tokens_for_text - min_doc_tokens

        # We need to preserve [E] and [/E] and some context around them
        # Calculate how many tokens we can keep before and after
        tokens_before_e = e_start_pos
        tokens_in_e = e_end_pos - e_start_pos + 1
        tokens_after_e = mention_len - e_end_pos - 1

        # If mention span with markers fits, center the truncation around it
        if tokens_in_e < mention_max_len:
            remaining_space = mention_max_len - tokens_in_e

            # Try to keep some context before and after [E] markers
            keep_before = min(tokens_before_e, remaining_space // 2)
            keep_after = min(tokens_after_e, remaining_space - keep_before)

            # Adjust keep_before if we have extra space
            if keep_after < tokens_after_e:
                keep_before = min(tokens_before_e, remaining_space - keep_after)

            # Extract the preserved tokens
            start_idx = e_start_pos - keep_before
            end_idx = e_end_pos + 1 + keep_after
            mention_tokens = mention_tokens[start_idx:end_idx]
        else:
            # Even the mention span is too long, truncate within it
            # Keep [E] at start and [/E] at end, truncate middle
            keep_start = mention_max_len // 2
            keep_end = mention_max_len - keep_start
            mention_tokens = (
                mention_tokens[e_start_pos:e_start_pos + keep_start] +
                mention_tokens[e_end_pos + 1 - keep_end:e_end_pos + 1]
            )

        # Now truncate doc_context to fit remaining space
        doc_max_len = max_tokens_for_text - len(mention_tokens)
        doc_tokens = tokenizer.encode(doc_context, add_special_tokens=False)
        if len(doc_tokens) > doc_max_len:
            doc_tokens = doc_tokens[:doc_max_len]

    # Manually construct the sequence: <s> doc_tokens </s> </s> mention_tokens </s>
    input_ids = (
        [tokenizer.bos_token_id] +
        doc_tokens +
        [tokenizer.eos_token_id] +
        [tokenizer.eos_token_id] +  # RoBERTa uses double </s> between sequences
        mention_tokens +
        [tokenizer.eos_token_id]
    )

    # Pad to max_length
    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)

    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    elif padding_length < 0:
        # Truncate if somehow we exceeded max_length
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


class PairsDataset(Dataset):
    """
    PyTorch Dataset for mention pairs with representations.

    Concatenates document_context + mention_context using text_pair,
    matching the original CL_COREF tokenization approach.
    """
    def __init__(self, pairs_df: pd.DataFrame, tokenizer, max_length=128):
        """
        Args:
            pairs_df: DataFrame with mention representations for both mentions
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization (default: 128)
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]

        # Get BOTH contexts: document (text) and mention (text_pair with [E] markers)
        doc_context1 = row['mention1_doc_context']
        mention_context1 = row['mention1_mention_context']

        doc_context2 = row['mention2_doc_context']
        mention_context2 = row['mention2_mention_context']

        # Use custom tokenization that preserves [E] markers
        tokenized1 = custom_tokenize_with_special_tokens(
            self.tokenizer,
            doc_context1,
            mention_context1,
            max_length=self.max_length
        )

        tokenized2 = custom_tokenize_with_special_tokens(
            self.tokenizer,
            doc_context2,
            mention_context2,
            max_length=self.max_length
        )

        # Convert label to float
        label = float(row['is_positive'])

        return tokenized1, tokenized2, label


class ValidationMentionsDataset(Dataset):
    """
    PyTorch Dataset for validation mentions (for clustering).

    Concatenates document_context + mention_context using text_pair,
    matching the original CL_COREF tokenization approach.
    """
    def __init__(self, mentions_df: pd.DataFrame, tokenizer, max_length=128):
        """
        Args:
            mentions_df: DataFrame with mention representations
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization (default: 128)
        """
        self.mentions_df = mentions_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.mentions_df)

    def __getitem__(self, idx):
        row = self.mentions_df.iloc[idx]

        # Get BOTH contexts: document and mention (with [E] markers)
        doc_context = row['document_context']
        mention_context = row['mention_context']

        # Use custom tokenization that preserves [E] markers
        tokenized = custom_tokenize_with_special_tokens(
            self.tokenizer,
            doc_context,
            mention_context,
            max_length=self.max_length
        )

        return idx, tokenized


def collate_fn_train(batch):
    """Collate function for training pairs."""
    tokenized1_list, tokenized2_list, labels = zip(*batch)

    # Convert to tensors and stack
    # Each tokenized dict has 'input_ids' and 'attention_mask' as lists
    input_ids1 = torch.tensor([t['input_ids'] for t in tokenized1_list])
    attention_mask1 = torch.tensor([t['attention_mask'] for t in tokenized1_list])

    input_ids2 = torch.tensor([t['input_ids'] for t in tokenized2_list])
    attention_mask2 = torch.tensor([t['attention_mask'] for t in tokenized2_list])

    labels = torch.tensor(labels, dtype=torch.float)

    return input_ids1, attention_mask1, input_ids2, attention_mask2, labels


def collate_fn_val(batch):
    """Collate function for validation mentions."""
    indices, tokenized_list = zip(*batch)

    # Convert to tensors and stack
    # Each tokenized dict has 'input_ids' and 'attention_mask' as lists
    input_ids = torch.tensor([t['input_ids'] for t in tokenized_list])
    attention_mask = torch.tensor([t['attention_mask'] for t in tokenized_list])

    indices = torch.tensor(indices, dtype=torch.long)

    return indices, input_ids, attention_mask


logger = logging.getLogger(__name__)


def filter_pairs_by_dataset(
    pairs_dict: pd.DataFrame,
    mentions_df: pd.DataFrame,
    dataset_name: str,
    training_mode: str = "single"
) -> pd.DataFrame:
    """
    Filter pairs to a specific dataset or all datasets except one.

    Args:
        pairs_dict: DataFrame with columns: dataset, mention1_id, mention2_id, is_positive, similarity
        mentions_df: DataFrame with mention_id, dataset_name, split, etc.
        dataset_name: Name of dataset (meaning depends on training_mode)
        training_mode: "single" (train on one dataset) or "all_but" (train on all except one)

    Returns:
        DataFrame with filtered pairs including split information
    """
    # Datasets with train splits
    DATASETS_WITH_TRAIN = ["ECBplus", "GVC", "HyperCoref", "NiDENT", "NP4E", "WECEng"]

    if training_mode == "single":
        logger.info(f"Training mode: single dataset - {dataset_name}")

        # Filter to specific dataset
        pairs_df = pairs_dict[pairs_dict['dataset'] == dataset_name].copy()

        if len(pairs_df) == 0:
            raise ValueError(f"No pairs found for dataset: {dataset_name}")

        logger.info(f"Loaded {len(pairs_df):,} pairs for {dataset_name}")

    elif training_mode == "all_but":
        logger.info(f"Training mode: all-but - excluding {dataset_name}")

        # Get all datasets except the excluded one
        participating_datasets = [d for d in DATASETS_WITH_TRAIN if d != dataset_name]
        logger.info(f"Participating datasets: {participating_datasets}")

        # Sample pairs from each participating dataset
        TARGET_TOTAL_PAIRS = 50000
        pairs_per_dataset = TARGET_TOTAL_PAIRS // len(participating_datasets)

        logger.info(f"Target: {TARGET_TOTAL_PAIRS:,} total pairs, ~{pairs_per_dataset:,} per dataset")

        sampled_pairs_list = []

        for dataset in participating_datasets:
            # Get all pairs for this dataset
            dataset_pairs = pairs_dict[pairs_dict['dataset'] == dataset].copy()

            if len(dataset_pairs) == 0:
                logger.warning(f"No pairs found for dataset: {dataset}, skipping")
                continue

            # Separate positive and negative pairs
            pos_pairs = dataset_pairs[dataset_pairs['is_positive'] == True]
            neg_pairs = dataset_pairs[dataset_pairs['is_positive'] == False]

            # Calculate target with 1:10 ratio
            target_pos = pairs_per_dataset // 11  # 1 out of 11 parts
            target_neg = pairs_per_dataset - target_pos  # 10 out of 11 parts

            # Sample (or take all if not enough)
            sampled_pos = pos_pairs.sample(n=min(target_pos, len(pos_pairs)), random_state=42)
            sampled_neg = neg_pairs.sample(n=min(target_neg, len(neg_pairs)), random_state=42)

            sampled_dataset_pairs = pd.concat([sampled_pos, sampled_neg], ignore_index=True)

            logger.info(f"  {dataset}: sampled {len(sampled_dataset_pairs):,} pairs "
                       f"({len(sampled_pos):,} pos, {len(sampled_neg):,} neg)")

            sampled_pairs_list.append(sampled_dataset_pairs)

        # Combine all sampled pairs
        pairs_df = pd.concat(sampled_pairs_list, ignore_index=True)
        logger.info(f"Total sampled pairs: {len(pairs_df):,}")

    else:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'single' or 'all_but'")

    # Join with mentions to get split information
    # Create mention lookup for split
    mention_split_lookup = mentions_df.set_index('mention_id')['split'].to_dict()

    # Assign split based on mention1 (assuming both mentions in pair are from same split)
    pairs_df['split'] = pairs_df['mention1_id'].map(mention_split_lookup)

    # Filter out pairs with missing split info
    pairs_df = pairs_df.dropna(subset=['split'])

    logger.info(f"Pairs by split:")
    for split in pairs_df['split'].unique():
        count = len(pairs_df[pairs_df['split'] == split])
        pos_count = len(pairs_df[(pairs_df['split'] == split) & pairs_df['is_positive']])
        neg_count = count - pos_count
        logger.info(f"  {split}: {count:,} total ({pos_count:,} positive, {neg_count:,} negative)")

    return pairs_df


def prepare_training_data(
    pairs_df: pd.DataFrame,
    mention_representations: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and validation data by joining pairs with mention representations.

    Args:
        pairs_df: DataFrame with mention1_id, mention2_id, is_positive, similarity, split
        mention_representations: DataFrame with mention_id, string_representation,
                                document_context, mention_context

    Returns:
        Tuple of (train_data, val_data) DataFrames with all necessary columns
    """
    logger.info("Preparing training and validation data")

    # Drop duplicate mention_ids (keep first occurrence)
    mention_representations_unique = mention_representations.drop_duplicates(subset='mention_id', keep='first')
    logger.info(f"Mention representations: {len(mention_representations):,} total, {len(mention_representations_unique):,} unique")

    # Create mention representation lookup
    mention_repr_dict = mention_representations_unique.set_index('mention_id').to_dict('index')

    # Separate train and validation pairs
    train_pairs = pairs_df[pairs_df['split'] == 'train'].copy()
    val_pairs = pairs_df[pairs_df['split'] == 'val'].copy()

    logger.info(f"Train pairs: {len(train_pairs):,}")
    logger.info(f"Val pairs: {len(val_pairs):,}")

    # Function to add mention representations to pairs
    def add_representations(df):
        # Add mention1 representations
        df['mention1_string_repr'] = df['mention1_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('string_representation')
        )
        df['mention1_doc_context'] = df['mention1_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('document_context')
        )
        df['mention1_mention_context'] = df['mention1_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('mention_context')
        )

        # Add mention2 representations
        df['mention2_string_repr'] = df['mention2_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('string_representation')
        )
        df['mention2_doc_context'] = df['mention2_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('document_context')
        )
        df['mention2_mention_context'] = df['mention2_id'].map(
            lambda x: mention_repr_dict.get(x, {}).get('mention_context')
        )

        # Drop rows with missing representations
        df = df.dropna(subset=[
            'mention1_string_repr', 'mention1_doc_context', 'mention1_mention_context',
            'mention2_string_repr', 'mention2_doc_context', 'mention2_mention_context'
        ])

        return df

    train_data = add_representations(train_pairs)
    val_data = add_representations(val_pairs)

    logger.info(f"After joining with representations:")
    logger.info(f"  Train data: {len(train_data):,} pairs")
    logger.info(f"  Val data: {len(val_data):,} pairs")

    return train_data, val_data


def prepare_validation_mentions(
    mentions_df: pd.DataFrame,
    mention_representations: pd.DataFrame,
    dataset_name: str,
    training_mode: str = "single"
) -> pd.DataFrame:
    """
    Prepare all validation mentions with their representations for evaluation.

    Args:
        mentions_df: DataFrame with mention metadata
        mention_representations: DataFrame with mention representations
        dataset_name: Name of dataset (meaning depends on training_mode)
        training_mode: "single" (validate on one dataset) or "all_but" (validate on all except one)

    Returns:
        DataFrame with all validation mentions and their representations
    """
    # Drop duplicate mention_ids (keep first occurrence)
    mention_representations_unique = mention_representations.drop_duplicates(subset='mention_id', keep='first')
    logger.info(f"Mention representations: {len(mention_representations):,} total, {len(mention_representations_unique):,} unique")

    if training_mode == "single":
        logger.info(f"Preparing validation mentions for {dataset_name}")

        # Filter to validation split and specific dataset
        val_mentions = mentions_df[
            (mentions_df['split'] == 'val') &
            (mentions_df['dataset_name'] == dataset_name)
        ].copy()

        logger.info(f"Validation mentions for {dataset_name}: {len(val_mentions):,}")

    elif training_mode == "all_but":
        logger.info(f"Preparing validation mentions for all-but-{dataset_name}")

        # Filter to validation split, excluding the specified dataset
        val_mentions = mentions_df[
            (mentions_df['split'] == 'val') &
            (mentions_df['dataset_name'] != dataset_name)
        ].copy()

        logger.info(f"Validation mentions (excluding {dataset_name}): {len(val_mentions):,}")

        # Log per-dataset statistics
        logger.info("Validation mentions by dataset:")
        for ds_name in sorted(val_mentions['dataset_name'].unique()):
            ds_count = len(val_mentions[val_mentions['dataset_name'] == ds_name])
            logger.info(f"  {ds_name}: {ds_count:,} mentions")

    else:
        raise ValueError(f"Invalid training_mode: {training_mode}. Must be 'single' or 'all_but'")

    # Join with representations
    val_mentions = val_mentions.merge(
        mention_representations_unique,
        on='mention_id',
        how='inner'
    )

    # Normalize columns with mixed types across datasets (int vs str) for parquet compatibility
    for col in ['topic_id', 'topic', 'subtopic', 'coref_chain']:
        if col in val_mentions.columns:
            val_mentions[col] = val_mentions[col].astype(str)

    logger.info(f"After joining with representations: {len(val_mentions):,} mentions")

    # Log statistics
    logger.info(f"Validation mentions by topic: {val_mentions['topic'].nunique()} unique topics")
    logger.info(f"Validation mentions by coref_chain: {val_mentions['coref_chain'].nunique()} unique chains")

    return val_mentions


def train_siamese_network(
    train_data: pd.DataFrame,
    val_mentions: pd.DataFrame,
    config: Dict,
    dataset_name: str,
    training_mode: str = "single"
) -> Dict:
    """
    Train siamese network using PyTorch Lightning.

    Note: Validation is done via clustering of val_mentions, not pairwise comparison.

    Args:
        train_data: Training pairs with representations
        val_mentions: All validation mentions with representations (for clustering evaluation)
        config: Training configuration dict
        dataset_name: Name of dataset (meaning depends on training_mode)
        training_mode: "single" or "all_but"

    Returns:
        Dict with training results and model checkpoint path
    """
    # Construct the effective dataset name for logging and checkpoints
    if training_mode == "all_but":
        effective_name = f"all-but-{dataset_name}"
    else:
        effective_name = dataset_name

    logger.info("="*80)
    logger.info(f"TRAINING SIAMESE NETWORK FOR {effective_name}")
    logger.info("="*80)

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Dataset: {effective_name}")
    logger.info(f"  Train pairs (for pairwise training): {len(train_data):,}")
    logger.info(f"    Positive: {train_data['is_positive'].sum():,}")
    logger.info(f"    Negative: {(~train_data['is_positive']).sum():,}")
    logger.info(f"  \nValidation approach: Clustering-based (not pairwise)")
    logger.info(f"    Validation mentions for clustering: {len(val_mentions):,}")
    logger.info(f"    Unique topics: {val_mentions['topic'].nunique()}")
    logger.info(f"    Unique coref chains: {val_mentions['coref_chain'].nunique()}")

    logger.info(f"\nModel configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Set random seed
    L.seed_everything(42)

    # Step 1: Create tokenizer
    logger.info("\nInitializing tokenizer and model...")
    special_tokens = [f"[{t}]" for t in config['special_tokens']] + [f"[/{t}]" for t in config['special_tokens']]
    tokenizer = create_tokenizer(config['language_model'], special_tokens)

    # Step 2: Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    max_length = config.get('max_length', 128)

    logger.info(f"Using max_length={max_length} for tokenization (document_context + mention_context)")
    logger.info("Tokenization uses text_pair: document_context truncated first, mention_context with [E] preserved")

    # Note: context_type parameter is no longer used, we always concatenate both contexts
    train_dataset = PairsDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = ValidationMentionsDataset(val_mentions, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=collate_fn_train,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_val,
        num_workers=0
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # Step 3: Create model
    logger.info("Initializing Siamese network...")
    model = create_siamese_model(config['language_model'], tokenizer)

    # Step 4: Create custom DataModule with validation mentions
    logger.info("Creating DataModule with validation wrapper...")
    datamodule = CorefDataModule(
        train_loader=train_loader,
        val_loader=val_loader,
        val_mentions_df=val_mentions,
        dataset_name=effective_name
    )
    datamodule.setup("fit")

    # Step 5: Create Lightning module with clustering-based validation
    lightning_config = {
        "model": {
            "language_model": config['language_model'],
            "learning_rate": config['learning_rate'],
            "margin": 0.7,  # Default margin for contrastive loss
            "clustering_threshold": config.get('similarity_threshold', 0.5)
        }
    }

    lightning_module = CorefLightningModule(model, lightning_config)

    # Step 6: Setup callbacks and loggers
    logger.info("Setting up callbacks and loggers...")

    # Model checkpoint
    checkpoint_dir = Path("data/06_models") / effective_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"siamese_model_{effective_name}_{{epoch:02d}}_{{val_f1_conll:.2f}}",
        monitor="val_f1_conll",
        mode='max',
        save_top_k=config.get('save_top_k', 1),
        save_last=True
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{effective_name}_{config['language_model']}"
    )

    # Weights & Biases logger
    # Create run name with dataset, epochs, and date
    current_date = datetime.now().strftime("%Y%m%d")
    num_epochs = config['num_epochs']
    run_name = f"{effective_name}_{config['language_model']}_ep{num_epochs}_{current_date}"

    wandb_logger = WandbLogger(
        project="cdcr-lexical-diversity",
        name=run_name,
        config=config,
        log_model=False
    )

    logger.info(f"W&B run name: {run_name}")

    # Step 7: Create trainer
    logger.info("Creating Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=config['num_epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        precision=config.get('precision', '32'),
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        log_every_n_steps=config.get('log_every_n_steps', 50),
        enable_checkpointing=True,
        enable_model_summary=True
    )

    # Step 8: Train model with clustering-based validation
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING WITH CLUSTERING-BASED VALIDATION")
    logger.info(f"Clustering threshold: {config.get('similarity_threshold', 0.5)}")
    logger.info("="*80 + "\n")

    trainer.fit(lightning_module, datamodule=datamodule)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)

    # Finish wandb run
    wandb.finish()

    # Return results
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"\nBest model saved to: {best_model_path}")

    return {
        "status": "completed",
        "dataset": effective_name,
        "training_mode": training_mode,
        "dataset_name": dataset_name,  # Original dataset name (single: trained on, all_but: excluded)
        "train_pairs": len(train_data),
        "val_mentions": len(val_mentions),
        "best_model_path": best_model_path,
        "best_metric_value": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None
    }
