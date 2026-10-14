"""
Pipeline for pairs selection using sentence-level similarities.

This pipeline implements the sentence-centric approach from sampling.py:
1. Extract sentences by topic (raw text, no [E]/[/E] markers)
2. Encode sentences with SBERT and compute pairwise similarities
3. Create mention pairs via product() from sentence pairs
4. Apply hard mining based on sentence similarities
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_sentences_by_topic,
    encode_sentences_by_topic,
    create_mention_pairs_from_sentences,
    compute_pairs_summary_stats,
    filter_pairs_by_similarity_quantiles,
    sample_balanced_pairs_per_dataset,
    select_hard_pairs_sentence_based,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the pairs selection pipeline.

    This pipeline:
    1. Prepares sentences grouped by topic (train split only)
    2. Encodes sentences with SBERT and computes sentence-pair similarities
    3. Creates mention pairs via Cartesian product of mentions in sentence pairs
       - Assigns sentence similarity to all mention pairs
    4. Computes summary statistics at dataset level
       - Count of pairs by (dataset, is_positive)
       - Similarity quantiles: 0.25, 0.50, 0.75 and mean
    5. Filters pairs by similarity quantiles (per dataset):
       - Positive pairs: keep similarity < 65th percentile
       - Negative pairs: keep similarity > 96th percentile
    6. Samples balanced pairs per dataset:
       - 4,545 positive pairs + 45,455 negative pairs = 50,000 total per dataset

    Returns:
        Pipeline: The Kedro pipeline
    """
    return pipeline(
        [
            # Node 1: Prepare sentences by topic
            node(
                func=prepare_sentences_by_topic,
                inputs={
                    "mentions_df": "mentions_for_pairs",
                    "data_path": "params:data_path",
                },
                outputs=["sentences_by_topic", "mentions_by_sentence"],
                name="prepare_sentences_by_topic_node",
                tags=["prepare", "pairs_selection"],
            ),
            # Node 2: Encode sentences with SBERT
            node(
                func=encode_sentences_by_topic,
                inputs={
                    "sentences_by_topic": "sentences_by_topic",
                    "mentions_by_sentence": "mentions_by_sentence",
                    "model_name": "params:sbert_model_name",
                    "similarity_batch_size": "params:similarity_batch_size",
                },
                outputs="sentence_pairs_with_similarity",
                name="encode_sentences_by_topic_node",
                tags=["encode", "pairs_selection"],
            ),
            # Node 3: Create mention pairs from sentence pairs
            # Uses PartitionedDataset to avoid OOM from concatenating 113M+ rows
            node(
                func=create_mention_pairs_from_sentences,
                inputs={
                    "sentence_pairs_dict": "sentence_pairs_with_similarity",
                    "mentions_by_sentence": "mentions_by_sentence",
                    "mentions_df": "mentions_for_pairs",
                },
                outputs="mention_pairs_from_sentences",
                name="create_mention_pairs_from_sentences_node",
                tags=["pairs", "pairs_selection"],
            ),
            # Node 4: Compute summary statistics at dataset level
            node(
                func=compute_pairs_summary_stats,
                inputs={
                    "mention_pairs_dict": "mention_pairs_from_sentences",
                },
                outputs="pairs_summary_stats",
                name="compute_pairs_summary_stats_node",
                tags=["stats", "pairs_selection"],
            ),
            # Node 5: Filter pairs by similarity quantiles
            node(
                func=filter_pairs_by_similarity_quantiles,
                inputs={
                    "mention_pairs_dict": "mention_pairs_from_sentences",
                },
                outputs="pairs_filtered_by_similarity_quantiles",
                name="filter_pairs_by_similarity_quantiles_node",
                tags=["filter", "pairs_selection"],
            ),
            # Node 6: Sample balanced pairs per dataset (4,545 positive + 45,455 negative)
            node(
                func=sample_balanced_pairs_per_dataset,
                inputs={
                    "filtered_pairs_dict": "pairs_filtered_by_similarity_quantiles",
                    "n_positive_per_dataset": "params:n_positive_per_dataset",
                    "n_negative_per_dataset": "params:n_negative_per_dataset",
                    "random_seed": "params:random_seed",
                },
                outputs="pairs_sampled_balanced",
                name="sample_balanced_pairs_per_dataset_node",
                tags=["sample", "pairs_selection"],
            ),
            # Node 7: DISABLED - Select hard pairs
            # Uncomment when ready to apply hard mining filters
            # node(
            #     func=select_hard_pairs_sentence_based,
            #     inputs={
            #         "mention_pairs_df": "mention_pairs_from_sentences",
            #         "additional_sampling": "params:additional_sampling",
            #     },
            #     outputs="selected_pairs",
            #     name="select_hard_pairs_node",
            #     tags=["select", "pairs_selection"],
            # ),
        ],
        tags="02_pairs_selection",
    )
