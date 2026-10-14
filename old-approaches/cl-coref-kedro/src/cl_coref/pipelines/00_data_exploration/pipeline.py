"""
Pipeline for exploring CDCR datasets.

This pipeline loads all mention data from all datasets and creates a unified
dataset for analysis and statistics.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_all_mentions_from_datasets,
    apply_topic_filters,
    compute_dataset_statistics,
    save_mentions_to_parquet,
    save_statistics_to_json,
    create_summary_table,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data exploration pipeline.

    This pipeline:
    1. Loads all entity and event mentions from all datasets
    2. Adds metadata (dataset_name, split, entity_style) to each mention
    3. Combines into a single DataFrame
    4. Applies topic filters (adds is_filtered column)
    5. Saves to Parquet format for efficient analysis
    6. Computes and saves statistics (with filtering breakdown)

    Returns:
        Pipeline: The Kedro pipeline
    """
    return pipeline(
        [
            # Load all mentions from all datasets
            node(
                func=load_all_mentions_from_datasets,
                inputs="params:data_path",
                outputs="all_mentions_raw",
                name="load_all_mentions_node",
                tags=["load", "data_exploration"],
            ),
            # Apply topic filters (adds is_filtered column)
            node(
                func=apply_topic_filters,
                inputs={
                    "df": "all_mentions_raw",
                    "topic_filters": "params:topic_filters",
                },
                outputs="all_mentions_df",
                name="apply_topic_filters_node",
                tags=["filter", "data_exploration"],
            ),
            # Create summary table
            node(
                func=create_summary_table,
                inputs="all_mentions_df",
                outputs="summary_table",
                name="create_summary_table_node",
                tags=["summary", "data_exploration"],
            ),
            # Compute statistics (includes filtering breakdown)
            node(
                func=compute_dataset_statistics,
                inputs="all_mentions_df",
                outputs="dataset_statistics",
                name="compute_statistics_node",
                tags=["statistics", "data_exploration"],
            ),
            # Save mentions to Parquet (with is_filtered column)
            node(
                func=save_mentions_to_parquet,
                inputs={
                    "df": "all_mentions_df",
                    "output_path": "params:output_parquet_path",
                    "partition_cols": "params:partition_cols",
                },
                outputs="parquet_file_path",
                name="save_parquet_node",
                tags=["save", "data_exploration"],
            ),
            # Save statistics to JSON
            node(
                func=save_statistics_to_json,
                inputs={
                    "stats": "dataset_statistics",
                    "output_path": "params:output_stats_path",
                },
                outputs="stats_file_path",
                name="save_statistics_node",
                tags=["save", "data_exploration"],
            ),
        ],
        tags="00_data_exploration",
    )
