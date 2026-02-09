"""
Pipeline for mentions preparation.

This pipeline prepares filtered mentions (all splits: train, val, test) by creating
string representations with document context and mention markers, ready for encoding
and pair selection in downstream pipelines.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_filtered_mentions,
    select_required_columns,
    compute_grouping_statistics,
    create_mention_groups,
    create_mention_string_representation,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the mentions preparation pipeline.

    This pipeline:
    1. Loads filtered mentions from all splits (is_filtered=True, train/val/test)
    2. Selects only required columns for downstream processing
    3. Computes statistics grouped by dataset -> entity_style -> topic
    4. Creates summary table of groups
    5. Creates string representations for mentions (8 sentences context + mention with [E]/[/E] markers)

    The string representations are ready for:
    - Encoding with SBERT or similar models
    - Tokenization for model training
    - Pair selection in pipeline 02

    Returns:
        Pipeline: The Kedro pipeline
    """
    return pipeline(
        [
            # Load filtered mentions (all splits: train, val, test)
            node(
                func=load_filtered_mentions,
                inputs="all_mentions_df",
                outputs="filtered_mentions",
                name="load_filtered_mentions_node",
                tags=["load", "mentions_preparation"],
            ),
            # Select required columns (includes sent_id and tokens_number for context creation)
            node(
                func=select_required_columns,
                inputs="filtered_mentions",
                outputs="mentions_for_pairs",
                name="select_required_columns_node",
                tags=["prepare", "mentions_preparation"],
            ),
            # Compute grouping statistics
            node(
                func=compute_grouping_statistics,
                inputs="mentions_for_pairs",
                outputs="grouping_statistics",
                name="compute_grouping_statistics_node",
                tags=["statistics", "mentions_preparation"],
            ),
            # Create mention groups summary
            node(
                func=create_mention_groups,
                inputs="mentions_for_pairs",
                outputs="mention_groups_summary",
                name="create_mention_groups_node",
                tags=["summary", "mentions_preparation"],
            ),
            # Create string representations for mentions
            node(
                func=create_mention_string_representation,
                inputs={
                    "df": "mentions_for_pairs",
                    "data_path": "params:data_path",
                    "special_mention_tokens": "params:special_mention_tokens",
                },
                outputs="mention_representations",
                name="create_mention_string_representation_node",
                tags=["representation", "mentions_preparation"],
            ),
        ],
        tags="01_mentions_preparation",
    )
