"""
Pipeline for training siamese network on coreference pairs.

This pipeline:
1. Filters pairs to a specific dataset (e.g., ECBplus)
2. Prepares training and validation data by joining with mention representations
3. Prepares all validation mentions for end-of-epoch evaluation
4. Trains the siamese network using PyTorch Lightning
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    filter_pairs_by_dataset,
    prepare_training_data,
    prepare_validation_mentions,
    train_siamese_network,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the training pipeline.

    This pipeline trains a siamese network for coreference resolution:
    1. Filters pairs to specific dataset
    2. Joins pairs with mention representations
    3. Splits into train/val
    4. Prepares validation mentions for clustering evaluation
    5. Trains model with Lightning

    Returns:
        Pipeline: The Kedro pipeline
    """
    return pipeline(
        [
            # Node 1: Filter pairs to specific dataset or all-but
            node(
                func=filter_pairs_by_dataset,
                inputs={
                    "pairs_dict": "pairs_sampled_balanced",
                    "mentions_df": "mentions_for_pairs",
                    "dataset_name": "params:training_dataset",
                    "training_mode": "params:training_mode",
                },
                outputs="dataset_filtered_pairs",
                name="filter_pairs_by_dataset_node",
                tags=["filter", "training"],
            ),
            # Node 2: Prepare training data with representations
            node(
                func=prepare_training_data,
                inputs={
                    "pairs_df": "dataset_filtered_pairs",
                    "mention_representations": "mention_representations",
                },
                outputs=["train_pairs_with_repr", "val_pairs_with_repr"],
                name="prepare_training_data_node",
                tags=["prepare", "training"],
            ),
            # Node 3: Prepare validation mentions for evaluation
            node(
                func=prepare_validation_mentions,
                inputs={
                    "mentions_df": "mentions_for_pairs",
                    "mention_representations": "mention_representations",
                    "dataset_name": "params:training_dataset",
                    "training_mode": "params:training_mode",
                },
                outputs="val_mentions_with_repr",
                name="prepare_validation_mentions_node",
                tags=["prepare", "training"],
            ),
            # Node 4: Train siamese network (validation via clustering)
            node(
                func=train_siamese_network,
                inputs={
                    "train_data": "train_pairs_with_repr",
                    "val_mentions": "val_mentions_with_repr",
                    "config": "params:training_config",
                    "dataset_name": "params:training_dataset",
                    "training_mode": "params:training_mode",
                },
                outputs="training_results",
                name="train_siamese_network_node",
                tags=["train", "training"],
            ),
        ],
        tags="03_training",
    )
