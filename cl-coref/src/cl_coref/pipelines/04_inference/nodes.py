"""
Nodes for inference pipeline.

This module implements test-time inference with subtopic-level clustering
and CoNLL file generation for evaluation.
"""

import logging
import os
import importlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_test_mentions_for_inference(
    mentions_df: pd.DataFrame,
    mention_representations: pd.DataFrame,
    test_datasets: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Prepare test mentions for all specified datasets.

    Args:
        mentions_df: DataFrame with mention metadata
        mention_representations: DataFrame with mention representations
        test_datasets: List of dataset names to prepare

    Returns:
        Dict mapping dataset_name to test mentions DataFrame with representations
    """
    logger.info(f"Preparing test mentions for {len(test_datasets)} datasets")

    # Drop duplicate mention_ids in representations
    mention_representations_unique = mention_representations.drop_duplicates(
        subset='mention_id', keep='first'
    )

    result = {}

    for dataset_name in test_datasets:
        # Filter to test split and specific dataset
        test_mentions = mentions_df[
            (mentions_df['split'] == 'test') &
            (mentions_df['dataset_name'] == dataset_name)
        ].copy()

        if len(test_mentions) == 0:
            logger.warning(f"No test mentions found for {dataset_name}")
            continue

        # Join with representations
        test_mentions = test_mentions.merge(
            mention_representations_unique,
            on='mention_id',
            how='inner'
        )

        logger.info(f"  {dataset_name}: {len(test_mentions):,} test mentions")

        # Log entity/event distribution
        if 'entity_style' in test_mentions.columns:
            n_entities = len(test_mentions[test_mentions['entity_style'] == 'entity'])
            n_events = len(test_mentions[test_mentions['entity_style'] == 'event'])
            logger.info(f"    Entities: {n_entities:,}, Events: {n_events:,}")

        result[dataset_name] = test_mentions

    return result


def run_inference_with_model(
    test_mentions_dict: Dict[str, pd.DataFrame],
    model_checkpoint_path: str,
    similarity_threshold: float,
    training_config: str
) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Run inference on test data using trained model.

    For each dataset and mode (Events, Entities, Mix):
    1. Filter mentions by entity_style
    2. Load trained model
    3. Encode mentions
    4. Cluster by subtopic independently
    5. Generate predicted cluster IDs

    Args:
        test_mentions_dict: Dict mapping dataset_name to test mentions DataFrame
        model_checkpoint_path: Path to trained model checkpoint
        similarity_threshold: Clustering threshold (cosine distance)
        training_config: Training configuration name (for output paths)

    Returns:
        Nested dict: {dataset_name: {mode: {subtopic_id: {mention_id: cluster_id}}}}
    """
    logger.info(f"Running inference with model: {model_checkpoint_path}")
    logger.info(f"Training config: {training_config}")
    logger.info(f"Similarity threshold: {similarity_threshold}")

    # Import required modules from CL_COREF
    import sys
    from pathlib import Path
    CL_COREF_PATH = Path(__file__).parents[5] / "CL_COREF"
    sys.path.insert(0, str(CL_COREF_PATH))
    from model import create_siamese_model, create_tokenizer

    # Import Lightning module dynamically (module name starts with number)
    lightning_module = importlib.import_module('cl_coref.pipelines.03_training.lightning_module')
    CorefLightningModule = lightning_module.CorefLightningModule

    # Import custom tokenization function from training
    training_nodes = importlib.import_module('cl_coref.pipelines.03_training.nodes')
    custom_tokenize_with_special_tokens = training_nodes.custom_tokenize_with_special_tokens

    # Load checkpoint to get hyperparameters
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    hparams = checkpoint.get('hyper_parameters', {})

    # Extract config from hyperparameters
    config = hparams.get('config', {})
    language_model = config.get('language_model', 'roberta-large')
    max_length = config.get('max_length', 128)

    logger.info(f"Reconstructing model: {language_model}")

    # Reconstruct the model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tokenizer with special tokens [E] and [/E]
    # NOTE: Need to add BOTH [E] and [/E] tokens with brackets
    tokenizer = create_tokenizer(language_model, special_tokens=["[E]", "[/E]"])
    siamese_model = create_siamese_model(language_model, tokenizer)

    # Verify the special tokens are added correctly
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    e_id = tokenizer.convert_tokens_to_ids('[E]')
    e_end_id = tokenizer.convert_tokens_to_ids('[/E]')
    logger.info(f"Special tokens: [E] = {e_id}, [/E] = {e_end_id}")

    # Create Lightning module and load state dict manually
    lightning_model = CorefLightningModule(model=siamese_model, config=config)

    # Load state dict with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = lightning_model.load_state_dict(
        checkpoint['state_dict'],
        strict=False
    )

    if missing_keys:
        logger.warning(f"Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

    lightning_model.eval()
    lightning_model.to(device)

    # Store tokenizer and model for encoding
    tokenizer_obj = tokenizer
    encoder = lightning_model.model

    logger.info(f"Model loaded successfully on {device}")

    results = {}

    # Process each dataset
    for dataset_name, test_mentions in test_mentions_dict.items():
        logger.info(f"\nProcessing dataset: {dataset_name}")

        dataset_results = {}

        # Process three modes: Events, Entities, Mix
        modes = ['Events', 'Entities', 'Mix']

        for mode in modes:
            logger.info(f"  Mode: {mode}")

            # Filter mentions by entity_style
            if mode == 'Events':
                filtered_mentions = test_mentions[
                    test_mentions['entity_style'] == 'event'
                ].copy()
            elif mode == 'Entities':
                filtered_mentions = test_mentions[
                    test_mentions['entity_style'] == 'entity'
                ].copy()
            else:  # Mix
                filtered_mentions = test_mentions.copy()

            if len(filtered_mentions) == 0:
                logger.warning(f"    No mentions for mode {mode}, skipping")
                continue

            logger.info(f"    Mentions: {len(filtered_mentions):,}")

            # Group by subtopic
            subtopic_results = {}

            subtopics = filtered_mentions['subtopic_id'].unique()
            logger.info(f"    Subtopics: {len(subtopics)}")

            for subtopic_id in tqdm(subtopics, desc=f"    Clustering subtopics"):
                subtopic_mentions = filtered_mentions[
                    filtered_mentions['subtopic_id'] == subtopic_id
                ].copy()

                if len(subtopic_mentions) < 2:
                    # Single mention = singleton cluster
                    subtopic_results[subtopic_id] = {
                        subtopic_mentions.iloc[0]['mention_id']: 0
                    }
                    continue

                # Extract contexts for encoding (need BOTH document_context and mention_context)
                document_contexts = subtopic_mentions['document_context'].tolist()
                mention_contexts = subtopic_mentions['mention_context'].tolist()
                mention_ids = subtopic_mentions['mention_id'].tolist()

                # Encode with trained model
                with torch.no_grad():
                    # Use custom tokenization that preserves [E] markers (same as training)
                    # Process each mention individually to ensure proper handling
                    batch_input_ids = []
                    batch_attention_mask = []

                    for doc_ctx, men_ctx in zip(document_contexts, mention_contexts):
                        tokenized = custom_tokenize_with_special_tokens(
                            tokenizer_obj,
                            doc_ctx,
                            men_ctx,
                            max_length=max_length
                        )
                        batch_input_ids.append(tokenized['input_ids'])
                        batch_attention_mask.append(tokenized['attention_mask'])

                    # Convert to tensors
                    input_ids = torch.tensor(batch_input_ids).to(device)
                    attention_mask = torch.tensor(batch_attention_mask).to(device)

                    # Encode - SiameseNetwork.forward expects positional args
                    embeddings = encoder(input_ids, attention_mask)
                    embeddings = embeddings.cpu().numpy()

                # Cluster embeddings
                clusters = cluster_embeddings(
                    embeddings,
                    threshold=similarity_threshold
                )

                # Map mention_id to cluster_id
                mention_to_cluster = {
                    mid: cid for mid, cid in zip(mention_ids, clusters)
                }

                subtopic_results[subtopic_id] = mention_to_cluster

            dataset_results[mode] = subtopic_results

        results[dataset_name] = dataset_results

    return results


def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Cluster embeddings using hierarchical clustering.

    Args:
        embeddings: (n_mentions, embedding_dim) array
        threshold: Distance threshold for clustering

    Returns:
        Array of cluster IDs (n_mentions,)
    """
    if len(embeddings) == 1:
        return np.array([0])

    # Compute cosine distances
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities

    # Clip distances to [0, 2] to handle floating point precision issues
    # (cosine distance is theoretically in [0, 2])
    distances = np.clip(distances, 0, 2)

    # Convert to condensed distance matrix
    condensed_dist = squareform(distances, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    return clusters - 1  # Convert to 0-indexed


def write_conll_files(
    test_mentions_dict: Dict[str, pd.DataFrame],
    predictions: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
    output_base_dir: str,
    training_config: str
):
    """
    Write CoNLL format key and response files for evaluation.

    Args:
        test_mentions_dict: Dict mapping dataset_name to test mentions DataFrame
        predictions: Nested dict with predicted clusters
        output_base_dir: Base output directory
        training_config: Training configuration name (e.g., "single-ECBplus", "mix")
    """
    logger.info(f"\nWriting CoNLL files to {output_base_dir}/{training_config}")

    output_dir = Path(output_base_dir) / training_config

    for dataset_name in predictions:
        logger.info(f"\nDataset: {dataset_name}")

        test_mentions = test_mentions_dict[dataset_name]
        dataset_predictions = predictions[dataset_name]

        for mode in dataset_predictions:
            logger.info(f"  Mode: {mode}")

            # Create output directory
            mode_dir = output_dir / dataset_name / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            # Filter mentions for this mode
            if mode == 'Events':
                filtered_mentions = test_mentions[
                    test_mentions['entity_style'] == 'event'
                ].copy()
            elif mode == 'Entities':
                filtered_mentions = test_mentions[
                    test_mentions['entity_style'] == 'entity'
                ].copy()
            else:  # Mix
                filtered_mentions = test_mentions.copy()

            if len(filtered_mentions) == 0:
                logger.warning(f"    No mentions for mode {mode}, skipping")
                continue

            # Sort by subtopic_id and mention_id for consistent ordering
            filtered_mentions = filtered_mentions.sort_values(
                ['subtopic_id', 'mention_id']
            ).reset_index(drop=True)

            # Get predictions for this mode
            mode_predictions = dataset_predictions[mode]

            # Write key.conll and response.conll
            key_path = mode_dir / 'key.conll'
            response_path = mode_dir / 'response.conll'

            with open(key_path, 'w') as key_file, \
                 open(response_path, 'w') as response_file:

                current_subtopic = None

                for _, row in filtered_mentions.iterrows():
                    subtopic_id = row['subtopic_id']
                    mention_id = row['mention_id']
                    true_cluster = row['coref_chain']

                    # Get predicted cluster
                    if subtopic_id in mode_predictions:
                        pred_cluster = mode_predictions[subtopic_id].get(mention_id, -1)
                    else:
                        pred_cluster = -1

                    # Write document header when subtopic changes
                    if current_subtopic != subtopic_id:
                        if current_subtopic is not None:
                            # Add blank line between documents
                            key_file.write('\n')
                            response_file.write('\n')

                        # Write document header
                        doc_id = f"{dataset_name}/{subtopic_id}"
                        header = f"#begin document ({doc_id}); part 000\n"
                        key_file.write(header)
                        response_file.write(header)

                        current_subtopic = subtopic_id

                    # Write mention lines (4 tab-separated columns for coval compatibility)
                    doc_id = f"{dataset_name}/{subtopic_id}"
                    coref_chain = row['coref_chain']
                    key_file.write(f"{doc_id}\t{mention_id}\t{coref_chain}\t({true_cluster})\n")
                    response_file.write(f"{doc_id}\t{mention_id}\t{coref_chain}\t({pred_cluster})\n")

            logger.info(f"    Written: {key_path}")
            logger.info(f"    Written: {response_path}")


def run_full_inference_pipeline(
    mentions_df: pd.DataFrame,
    mention_representations: pd.DataFrame,
    training_config: str,
    model_checkpoint_path: str,
    similarity_threshold: float,
    output_base_dir: str
) -> dict:
    """
    Run complete inference pipeline for a training configuration.

    Args:
        mentions_df: DataFrame with all mention metadata
        mention_representations: DataFrame with mention representations
        training_config: Training configuration (e.g., "single-ECBplus", "mix")
        model_checkpoint_path: Path to trained model checkpoint
        similarity_threshold: Clustering threshold
        output_base_dir: Base directory for output files

    Returns:
        Dict with summary statistics
    """
    logger.info("="*80)
    logger.info(f"INFERENCE PIPELINE - {training_config}")
    logger.info("="*80)

    # Determine which datasets to evaluate
    # All 8 datasets can be evaluated in test
    ALL_DATASETS = [
        'ECBplus', 'ECBplusMETAm', 'GVC', 'HyperCoref',
        'NiDENT', 'NP4E', 'NewsWCL50', 'WECEng'
    ]
    test_datasets = ALL_DATASETS

    logger.info(f"Test datasets: {test_datasets}")

    # Step 1: Prepare test mentions
    test_mentions_dict = prepare_test_mentions_for_inference(
        mentions_df,
        mention_representations,
        test_datasets
    )

    # Step 2: Run inference
    predictions = run_inference_with_model(
        test_mentions_dict,
        model_checkpoint_path,
        similarity_threshold,
        training_config
    )

    # Step 3: Write CoNLL files
    write_conll_files(
        test_mentions_dict,
        predictions,
        output_base_dir,
        training_config
    )

    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)

    return {
        "training_config": training_config,
        "datasets_evaluated": list(test_mentions_dict.keys()),
        "total_datasets": len(test_mentions_dict)
    }
