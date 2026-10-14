"""
Validation wrapper for clustering-based evaluation.

This module provides a lightweight wrapper around validation mentions
to enable clustering-based validation without requiring the full CDCR_Dataset structure.
"""

import pandas as pd
from typing import Dict
from collections import defaultdict


class SimpleMention:
    """
    Simplified Mention class that mimics the structure expected by evaluation functions.
    """
    def __init__(self, row: pd.Series, dataset_name: str):
        self.mention_id = row['mention_id']
        self.topic_id = str(row['topic'])
        self.subtopic_id = row.get('subtopic', row['topic'])  # Fallback to topic if no subtopic
        self.coref_chain = row['coref_chain']
        self.mention_head = row.get('mention_head', '')
        self.source_path = f"data/{dataset_name}/val"  # Mock path for dataset prefix extraction

        # These are needed for evaluation but may not be in our DataFrame
        self.mention_wo_stopwords = row.get('mention_wo_stopwords', [])
        self.mention_wordset = row.get('mention_wordset', set())


class ValidationDatasetWrapper:
    """
    Wrapper that mimics CDCR_Dataset structure for validation.

    This allows the existing evaluation functions to work with our DataFrame-based
    validation mentions without requiring the full CDCR_Dataset infrastructure.
    """
    def __init__(self, val_mentions_df: pd.DataFrame, dataset_name: str):
        """
        Args:
            val_mentions_df: DataFrame with validation mentions and their representations
            dataset_name: Name of the dataset (e.g., "ECBplus")
        """
        self.name = dataset_name
        self.mentions = {}  # Dict[mention_id, SimpleMention]
        self.mentions_by_topic = defaultdict(list)
        self.gold_clusters_df = None

        self._build_from_dataframe(val_mentions_df, dataset_name)

    def _build_from_dataframe(self, df: pd.DataFrame, dataset_name: str):
        """Build mention structures from DataFrame."""
        mentions_gold_topic_repr = []

        for idx, row in df.iterrows():
            mention_id = row['mention_id']

            # Create SimpleMention object
            mention = SimpleMention(row, dataset_name)
            self.mentions[mention_id] = mention

            # Group by topic
            topic_id = str(row['topic'])
            self.mentions_by_topic[topic_id].append(mention_id)

            # Build gold cluster representation
            mentions_gold_topic_repr.append({
                "mention_id": mention_id,
                "subtopic_id": mention.subtopic_id,
                "coref_chain": row['coref_chain'],
                "mention_type": row.get('entity_style', 'unknown'),  # Use entity_style as mention_type
                "mention_head": row.get('mention_head', ''),
                "mention_wo_stopwords": mention.mention_wo_stopwords,
                "mention_wordset": mention.mention_wordset
            })

        # Create gold clusters DataFrame
        gold_clusters_df = pd.DataFrame(mentions_gold_topic_repr)
        gold_clusters_df["gold_cluster"] = pd.Categorical(gold_clusters_df["coref_chain"]).codes
        self.gold_clusters_df = gold_clusters_df
