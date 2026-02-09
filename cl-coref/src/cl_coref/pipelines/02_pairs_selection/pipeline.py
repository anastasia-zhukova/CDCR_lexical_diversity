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
       - Reports statistics by dataset and entity_style

    Note: Node 4 (hard mining) is excluded to see total pair counts first.

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
            # This will output statistics by dataset and entity_style
            node(
                func=create_mention_pairs_from_sentences,
                inputs={
                    "sentence_pairs_df": "sentence_pairs_with_similarity",
                    "mentions_by_sentence": "mentions_by_sentence",
                    "mentions_df": "mentions_for_pairs",
                },
                outputs="mention_pairs_from_sentences",
                name="create_mention_pairs_from_sentences_node",
                tags=["pairs", "pairs_selection"],
            ),
            # Node 4: DISABLED - Select hard pairs
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
