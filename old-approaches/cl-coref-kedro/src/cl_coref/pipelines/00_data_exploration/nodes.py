"""
Nodes for exploring CDCR datasets.

This module contains pure functions for loading and unifying all mention data
from multiple datasets into a single analyzable format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_all_mentions_from_datasets(
    data_path: str,
) -> pd.DataFrame:
    """
    Load all entity and event mentions from all datasets into a unified DataFrame.

    This function:
    1. Recursively finds all entity_mentions.json and event_mentions.json files
    2. Extracts metadata from the file path:
       - dataset_name: from parent directory (e.g., "ECBplus")
       - split: from immediate parent directory (e.g., "train", "val", "test")
       - entity_style: "entity" or "event" based on filename
    3. Adds these fields to each mention record
    4. Combines all mentions into a single DataFrame

    Args:
        data_path: Base path to datasets (e.g., "cl-coref/data/01_raw/cdcr_datasets")

    Returns:
        DataFrame with all mentions and added metadata columns:
        - dataset_name: Name of the dataset (str)
        - split: Train/val/test split (str)
        - entity_style: "entity" or "event" (str)
        - All original fields from the JSON
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")

    logger.info(f"Scanning for mention files in: {data_path}")

    all_mentions = []
    file_count = 0

    # Find all entity_mentions.json and event_mentions.json files
    for json_file in data_path.rglob("*_mentions.json"):
        # Extract metadata from path
        # Path structure: .../dataset_name/split/entity_mentions.json
        filename = json_file.name
        split_dir = json_file.parent.name
        dataset_dir = json_file.parent.parent.name

        # Determine entity_style from filename
        if filename == "entity_mentions.json":
            entity_style = "entity"
        elif filename == "event_mentions.json":
            entity_style = "event"
        else:
            logger.warning(f"Unexpected filename: {filename}, skipping")
            continue

        # Validate split
        if split_dir not in ["train", "val", "test"]:
            logger.warning(f"Unexpected split directory: {split_dir}, skipping {json_file}")
            continue

        logger.info(f"Loading: {dataset_dir}/{split_dir}/{filename}")

        try:
            with open(json_file, 'r') as f:
                mentions = json.load(f)

            # Add metadata to each mention
            for mention in mentions:
                mention["dataset_name"] = dataset_dir
                mention["split"] = split_dir
                mention["entity_style"] = entity_style

            all_mentions.extend(mentions)
            file_count += 1

            logger.info(f"  Loaded {len(mentions):,} mentions from {dataset_dir}/{split_dir}/{entity_style}")

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"✓ Loaded {file_count} files with {len(all_mentions):,} total mentions")

    # Convert to DataFrame
    if not all_mentions:
        logger.warning("No mentions loaded - returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame(all_mentions)

    logger.info(f"✓ Created DataFrame with shape: {df.shape}")
    logger.info(f"  Datasets: {df['dataset_name'].nunique()}")
    logger.info(f"  Splits: {df['split'].unique().tolist()}")
    logger.info(f"  Entity styles: {df['entity_style'].value_counts().to_dict()}")

    return df


def apply_topic_filters(
    df: pd.DataFrame,
    topic_filters: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Apply topic filters to the unified mentions DataFrame.

    Adds an `is_filtered` column:
    - True: Mention passes the filter (should be kept for analysis)
    - False: Mention is filtered out (excluded from analysis)

    For datasets not in topic_filters, all mentions are kept (is_filtered=True).
    For datasets in topic_filters, only mentions with matching topics are kept.

    Args:
        df: DataFrame with all mentions
        topic_filters: Dict mapping dataset_name to list of topics to keep
                      Example: {"HyperCoref": ["us", "politics"], "WECEng": ["Earthquake"]}

    Returns:
        DataFrame with added `is_filtered` column
    """
    if df.empty:
        logger.warning("Empty DataFrame - returning without filtering")
        df["is_filtered"] = True
        return df

    logger.info("Applying topic filters...")

    # Initialize all as filtered (kept)
    df["is_filtered"] = True

    # Track statistics
    total_mentions = len(df)
    filtered_out_count = 0

    # Apply filters per dataset
    for dataset_name, topics_to_keep in topic_filters.items():
        if dataset_name not in df["dataset_name"].values:
            logger.warning(f"Dataset '{dataset_name}' in filters but not found in data")
            continue

        # Get mentions for this dataset
        dataset_mask = df["dataset_name"] == dataset_name
        dataset_mentions = df[dataset_mask]
        original_count = len(dataset_mentions)

        # Mark mentions that DON'T match the filter as filtered out
        if "topic" in df.columns:
            # Check which mentions are NOT in the allowed topics
            filter_out_mask = dataset_mask & (~df["topic"].isin(topics_to_keep))
            df.loc[filter_out_mask, "is_filtered"] = False

            filtered_out = filter_out_mask.sum()
            kept = original_count - filtered_out
            filtered_out_count += filtered_out

            logger.info(
                f"  {dataset_name}: {original_count:,} → {kept:,} mentions "
                f"({kept/original_count*100:.1f}% kept, {filtered_out:,} filtered out)"
            )
            logger.info(f"    Keeping topics: {', '.join(topics_to_keep[:5])}{'...' if len(topics_to_keep) > 5 else ''}")
        else:
            logger.warning(f"  {dataset_name}: No 'topic' column found, skipping filter")

    # Log datasets without filters
    datasets_with_filters = set(topic_filters.keys())
    all_datasets = set(df["dataset_name"].unique())
    unfiltered_datasets = all_datasets - datasets_with_filters

    if unfiltered_datasets:
        logger.info(f"  Datasets without filters (all kept): {', '.join(sorted(unfiltered_datasets))}")

    # Summary
    kept_count = df["is_filtered"].sum()
    logger.info(
        f"✓ Filtering complete: {total_mentions:,} total → "
        f"{kept_count:,} kept ({kept_count/total_mentions*100:.1f}%), "
        f"{filtered_out_count:,} filtered out ({filtered_out_count/total_mentions*100:.1f}%)"
    )

    return df


def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the unified mentions dataset.

    Computes stats for both all mentions and filtered mentions (if is_filtered column exists).

    Args:
        df: DataFrame with all mentions

    Returns:
        Dictionary with statistics:
        - total_mentions: Total number of mentions (all)
        - filtered_mentions: Number of kept mentions (is_filtered=True)
        - discarded_mentions: Number of filtered out mentions (is_filtered=False)
        - by_dataset: Stats per dataset (with filtered breakdown)
        - by_split: Stats per split
        - by_entity_style: Stats per entity/event (with filtered breakdown)
        - cluster_stats: Coreference cluster statistics
    """
    if df.empty:
        logger.warning("Empty DataFrame - returning empty statistics")
        return {
            "total_mentions": 0,
            "filtered_mentions": 0,
            "discarded_mentions": 0,
            "by_dataset": {},
            "by_split": {},
            "by_entity_style": {},
            "by_entity_style_filtered": {},
            "cluster_stats": {}
        }

    logger.info("Computing dataset statistics...")

    # Check if filtering was applied
    has_filter = "is_filtered" in df.columns

    stats = {
        "total_mentions": int(len(df)),
        "total_files": int(len(df.groupby(["dataset_name", "split", "entity_style"]))),
    }

    # Add filtering stats if available
    if has_filter:
        filtered_df = df[df["is_filtered"]]
        discarded_df = df[~df["is_filtered"]]

        stats["filtered_mentions"] = int(len(filtered_df))
        stats["discarded_mentions"] = int(len(discarded_df))
        stats["filtering_applied"] = True
    else:
        stats["filtered_mentions"] = int(len(df))
        stats["discarded_mentions"] = 0
        stats["filtering_applied"] = False

    # Stats by dataset
    stats["by_dataset"] = {}
    for dataset_name in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset_name]

        dataset_stats = {
            "total_mentions": int(len(dataset_df)),
            "entity_mentions": int(len(dataset_df[dataset_df["entity_style"] == "entity"])),
            "event_mentions": int(len(dataset_df[dataset_df["entity_style"] == "event"])),
            "splits": dataset_df["split"].unique().tolist(),
            "num_topics": int(dataset_df["topic"].nunique()) if "topic" in dataset_df.columns else 0,
        }

        # Add filtering breakdown if available
        if has_filter:
            filtered_dataset = dataset_df[dataset_df["is_filtered"]]
            discarded_dataset = dataset_df[~dataset_df["is_filtered"]]

            dataset_stats["filtered_mentions"] = int(len(filtered_dataset))
            dataset_stats["discarded_mentions"] = int(len(discarded_dataset))
            dataset_stats["filtered_entity_mentions"] = int(len(filtered_dataset[filtered_dataset["entity_style"] == "entity"]))
            dataset_stats["filtered_event_mentions"] = int(len(filtered_dataset[filtered_dataset["entity_style"] == "event"]))
            dataset_stats["discarded_entity_mentions"] = int(len(discarded_dataset[discarded_dataset["entity_style"] == "entity"]))
            dataset_stats["discarded_event_mentions"] = int(len(discarded_dataset[discarded_dataset["entity_style"] == "event"]))

        stats["by_dataset"][dataset_name] = dataset_stats

    # Stats by split
    stats["by_split"] = {}
    for split in sorted(df["split"].unique()):
        split_df = df[df["split"] == split]
        split_stats = {
            "total_mentions": int(len(split_df)),
            "num_datasets": int(split_df["dataset_name"].nunique()),
        }

        if has_filter:
            split_stats["filtered_mentions"] = int((split_df["is_filtered"]).sum())
            split_stats["discarded_mentions"] = int((~split_df["is_filtered"]).sum())

        stats["by_split"][split] = split_stats

    # Stats by entity style - convert numpy int64 to Python int
    stats["by_entity_style"] = {k: int(v) for k, v in df["entity_style"].value_counts().to_dict().items()}

    # Stats by entity style for filtered mentions only
    if has_filter:
        filtered_df = df[df["is_filtered"]]
        discarded_df = df[~df["is_filtered"]]

        stats["by_entity_style_filtered"] = {
            "kept": {k: int(v) for k, v in filtered_df["entity_style"].value_counts().to_dict().items()},
            "discarded": {k: int(v) for k, v in discarded_df["entity_style"].value_counts().to_dict().items()}
        }

    # Cluster statistics (if coref_chain exists)
    if "coref_chain" in df.columns:
        cluster_sizes = df.groupby(["dataset_name", "split", "coref_chain"]).size()

        stats["cluster_stats"] = {
            "total_clusters": int(len(cluster_sizes)),
            "singletons": int((cluster_sizes == 1).sum()),
            "non_singletons": int((cluster_sizes > 1).sum()),
            "avg_cluster_size": float(cluster_sizes.mean()),
            "max_cluster_size": int(cluster_sizes.max()),
            "min_cluster_size": int(cluster_sizes.min()),
        }

        # Estimate training pairs
        positive_pairs = sum(n * (n - 1) // 2 for n in cluster_sizes if n > 1)
        negative_pairs = positive_pairs * 2  # Rough estimate

        stats["cluster_stats"]["estimated_positive_pairs"] = int(positive_pairs)
        stats["cluster_stats"]["estimated_negative_pairs"] = int(negative_pairs)
        stats["cluster_stats"]["estimated_total_pairs"] = int(positive_pairs + negative_pairs)

    # Log summary
    logger.info(f"Statistics computed:")
    logger.info(f"  Total mentions: {stats['total_mentions']:,}")
    logger.info(f"  Datasets: {len(stats['by_dataset'])}")
    logger.info(f"  Splits: {list(stats['by_split'].keys())}")
    if "cluster_stats" in stats and stats["cluster_stats"]:
        logger.info(f"  Clusters: {stats['cluster_stats']['total_clusters']:,}")
        logger.info(f"  Estimated pairs: ~{stats['cluster_stats'].get('estimated_total_pairs', 0):,}")

    return stats


def save_mentions_to_parquet(
    df: pd.DataFrame,
    output_path: str,
    partition_cols: List[str] = None
) -> str:
    """
    Save unified mentions DataFrame to Parquet format.

    Parquet is chosen because:
    - Efficient columnar storage for analytics
    - Native support for nested lists (tokens_text, mention_context)
    - Built-in compression (smaller than JSON)
    - Fast filtering and column selection
    - Compatible with pandas, dask, spark

    Args:
        df: DataFrame with all mentions
        output_path: Path to save the parquet file
        partition_cols: Optional columns to partition by (e.g., ["dataset_name", "split"])

    Returns:
        Path to saved file
    """
    if df.empty:
        logger.warning("Empty DataFrame - not saving")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(df):,} mentions to: {output_path}")

    # Save to parquet
    if partition_cols:
        logger.info(f"  Partitioning by: {partition_cols}")
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            partition_cols=partition_cols,
            index=False
        )
    else:
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

    # Report file size
    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved to {output_path} ({size_mb:.2f} MB)")
    else:
        # Partitioned output is a directory
        total_size = sum(f.stat().st_size for f in output_path.rglob("*.parquet"))
        size_mb = total_size / (1024 * 1024)
        logger.info(f"✓ Saved to {output_path}/ ({size_mb:.2f} MB, partitioned)")

    return str(output_path)


def save_statistics_to_json(
    stats: Dict[str, Any],
    output_path: str
) -> str:
    """
    Save statistics dictionary to JSON for easy inspection.

    Args:
        stats: Statistics dictionary from compute_dataset_statistics
        output_path: Path to save JSON file

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving statistics to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"✓ Saved statistics ({size_kb:.2f} KB)")

    return str(output_path)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing mentions per dataset/split/entity_style.

    Args:
        df: DataFrame with all mentions

    Returns:
        Summary DataFrame with counts
    """
    if df.empty:
        return pd.DataFrame()

    logger.info("Creating summary table...")

    # Create pivot-like summary
    summary = df.groupby(["dataset_name", "split", "entity_style"]).size().reset_index(name="mention_count")

    # Add total per dataset
    dataset_totals = df.groupby("dataset_name").size().reset_index(name="total_mentions")
    summary = summary.merge(dataset_totals, on="dataset_name", how="left")

    # Add cluster info if available
    if "coref_chain" in df.columns:
        cluster_counts = df.groupby(["dataset_name", "split", "entity_style"])["coref_chain"].nunique().reset_index(name="num_clusters")
        summary = summary.merge(cluster_counts, on=["dataset_name", "split", "entity_style"], how="left")

    # Sort by dataset and split
    summary = summary.sort_values(["dataset_name", "split", "entity_style"])

    logger.info(f"✓ Created summary table with {len(summary)} rows")

    return summary
