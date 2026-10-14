"""
Nodes for pairs creation pipeline.

This module contains functions for organizing filtered mentions into groups
for pair generation: by dataset, entity_style, and topic.

It also creates string representations for mentions that will be tokenized later.
"""

import logging
import os
from collections import namedtuple, defaultdict
from typing import Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class SanitizedException(Exception):
    """
    Custom exception that sanitizes bracket characters to prevent Rich markup errors.

    Converts (E) and (/E) tokens to <E> and </E> in all error messages to avoid
    Rich markup interpretation.
    """
    def __init__(self, message: str):
        # Sanitize brackets to angle brackets which are safe for Rich
        sanitized = str(message).replace('[', '<').replace(']', '>')
        super().__init__(sanitized)
        # Store sanitized message
        self.sanitized_message = sanitized

    def __str__(self):
        # Ensure string representation is also sanitized
        return self.sanitized_message

    def __repr__(self):
        # Also sanitize repr
        return f"SanitizedException({self.sanitized_message!r})"


def load_filtered_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load filtered mentions from all splits (train, val, test).

    Filters the unified mentions DataFrame to only mentions where is_filtered=True
    (passed topic filters). Includes all splits since the model needs representations
    for training, validation, and testing.

    Args:
        df: Unified mentions DataFrame from 00_data_exploration pipeline

    Returns:
        DataFrame with filtered mentions from all splits
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()

    logger.info("Loading filtered mentions from all splits...")
    logger.info(f"  Input: {len(df):,} total mentions")

    # Check required columns
    if "is_filtered" not in df.columns:
        logger.error("Column 'is_filtered' not found - run 00_data_exploration pipeline first")
        raise ValueError("Missing 'is_filtered' column")

    if "split" not in df.columns:
        logger.error("Column 'split' not found")
        raise ValueError("Missing 'split' column")

    # Filter to mentions that passed topic filters (all splits)
    filtered_df = df[df["is_filtered"]]

    logger.info(f"  Filtered: {len(filtered_df):,} mentions (is_filtered=True)")
    logger.info(f"    Datasets: {filtered_df['dataset_name'].nunique()}")
    logger.info(f"    Splits breakdown:")
    for split in sorted(filtered_df['split'].unique()):
        split_count = (filtered_df['split'] == split).sum()
        logger.info(f"      {split}: {split_count:,} mentions")
    logger.info(f"    Entity style: {(filtered_df['entity_style'] == 'entity').sum():,}")
    logger.info(f"    Event style: {(filtered_df['entity_style'] == 'event').sum():,}")

    if len(filtered_df) == 0:
        logger.warning("No filtered mentions found!")

    return filtered_df


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the columns needed for pair generation and string representation.

    Keeps only:
    - coref_chain: Coreference cluster ID
    - mention_id: Unique mention identifier
    - is_singleton: Whether mention is a singleton
    - conll_doc_key: Document key
    - topic_id: Topic ID
    - subtopic_id: Subtopic ID
    - dataset_name: Dataset name (added in 00_data_exploration)
    - split: Data split (should only be 'train' at this point)
    - entity_style: 'entity' or 'event' (added in 00_data_exploration)
    - topic: Topic name/description
    - sent_id: Sentence ID (needed for context creation)
    - tokens_number: Token positions in sentence (needed for placing [E] [/E] markers)

    Args:
        df: Filtered train mentions DataFrame

    Returns:
        DataFrame with only required columns
    """
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()

    logger.info("Selecting required columns...")

    # Define required columns
    required_cols = [
        "coref_chain",
        "mention_id",
        "is_singleton",
        "conll_doc_key",
        "topic_id",
        "subtopic_id",
        "dataset_name",
        "split",
        "entity_style",
        "topic",
        "sent_id",
        "tokens_number",
    ]

    # Check which columns are missing
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Select columns
    selected_df = df[required_cols].copy()

    logger.info(f"  Selected {len(required_cols)} columns from {len(selected_df):,} mentions")
    logger.info(f"    Columns: {', '.join(required_cols[:6])}{'...' if len(required_cols) > 6 else ''}")

    # Log unique values for grouping keys
    logger.info(f"  Grouping dimensions:")
    logger.info(f"    Datasets: {selected_df['dataset_name'].nunique()}")
    logger.info(f"    Entity styles: {selected_df['entity_style'].nunique()}")
    logger.info(f"    Topics: {selected_df['topic'].nunique()}")
    logger.info(f"    Splits: {selected_df['split'].unique().tolist()} (should only be 'train')")

    return selected_df


def compute_grouping_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics grouped by dataset, entity_style, and topic.

    Creates hierarchical grouping:
    - Level 1: Dataset
      - Level 2: Entity style (entity/event)
        - Level 3: Topic

    Args:
        df: DataFrame with selected columns for pair generation

    Returns:
        Dictionary with hierarchical grouping statistics
    """
    if df.empty:
        logger.warning("Empty DataFrame - returning empty statistics")
        return {
            "total_mentions": 0,
            "total_datasets": 0,
            "by_dataset": {}
        }

    logger.info("Computing grouping statistics...")

    stats = {
        "total_mentions": int(len(df)),
        "total_datasets": int(df["dataset_name"].nunique()),
        "total_splits": df["split"].unique().tolist(),
    }

    # Validate only train split
    if stats["total_splits"] != ["train"]:
        logger.warning(f"Expected only 'train' split, but found: {stats['total_splits']}")

    # Group by dataset -> entity_style -> topic
    stats["by_dataset"] = {}

    for dataset_name in sorted(df["dataset_name"].unique()):
        dataset_df = df[df["dataset_name"] == dataset_name]

        dataset_stats = {
            "total_mentions": int(len(dataset_df)),
            "entity_styles": {}
        }

        # Group by entity style within this dataset
        for entity_style in sorted(dataset_df["entity_style"].unique()):
            style_df = dataset_df[dataset_df["entity_style"] == entity_style]

            style_stats = {
                "total_mentions": int(len(style_df)),
                "num_topics": int(style_df["topic"].nunique()),
                "topics": {}
            }

            # Group by topic within this entity style
            for topic in sorted(style_df["topic"].unique()):
                topic_df = style_df[style_df["topic"] == topic]

                # Compute cluster statistics for this group
                cluster_sizes = topic_df.groupby("coref_chain").size()

                topic_stats = {
                    "mentions": int(len(topic_df)),
                    "clusters": int(len(cluster_sizes)),
                    "singletons": int((cluster_sizes == 1).sum()),
                    "non_singletons": int((cluster_sizes > 1).sum()),
                    "avg_cluster_size": float(cluster_sizes.mean()),
                    "max_cluster_size": int(cluster_sizes.max()),
                    "min_cluster_size": int(cluster_sizes.min()),
                }

                # Estimate pairs for this topic
                positive_pairs = sum(n * (n - 1) // 2 for n in cluster_sizes if n > 1)
                negative_pairs = positive_pairs * 2  # Rough estimate

                topic_stats["estimated_positive_pairs"] = int(positive_pairs)
                topic_stats["estimated_negative_pairs"] = int(negative_pairs)
                topic_stats["estimated_total_pairs"] = int(positive_pairs + negative_pairs)

                style_stats["topics"][topic] = topic_stats

            # Aggregate entity style stats
            style_stats["total_clusters"] = sum(
                t["clusters"] for t in style_stats["topics"].values()
            )
            style_stats["total_singletons"] = sum(
                t["singletons"] for t in style_stats["topics"].values()
            )
            style_stats["estimated_total_pairs"] = sum(
                t["estimated_total_pairs"] for t in style_stats["topics"].values()
            )

            dataset_stats["entity_styles"][entity_style] = style_stats

        # Aggregate dataset stats
        dataset_stats["total_clusters"] = sum(
            style["total_clusters"] for style in dataset_stats["entity_styles"].values()
        )
        dataset_stats["total_pairs"] = sum(
            style["estimated_total_pairs"] for style in dataset_stats["entity_styles"].values()
        )

        stats["by_dataset"][dataset_name] = dataset_stats

    # Overall aggregates
    stats["total_clusters"] = sum(
        ds["total_clusters"] for ds in stats["by_dataset"].values()
    )
    stats["total_estimated_pairs"] = sum(
        ds["total_pairs"] for ds in stats["by_dataset"].values()
    )

    # Log summary
    logger.info(f"✓ Grouping statistics computed:")
    logger.info(f"  Total mentions: {stats['total_mentions']:,}")
    logger.info(f"  Total clusters: {stats['total_clusters']:,}")
    logger.info(f"  Estimated pairs: ~{stats['total_estimated_pairs']:,}")
    logger.info(f"  Datasets: {stats['total_datasets']}")

    # Log breakdown by dataset
    for dataset_name, ds_stats in stats["by_dataset"].items():
        logger.info(f"  {dataset_name}:")
        for entity_style, style_stats in ds_stats["entity_styles"].items():
            logger.info(
                f"    {entity_style}: {style_stats['total_mentions']:,} mentions, "
                f"{style_stats['num_topics']} topics, ~{style_stats['estimated_total_pairs']:,} pairs"
            )

    return stats


def create_mention_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame showing the grouping hierarchy.

    Creates a table with one row per (dataset, entity_style, topic) group,
    showing counts for each group.

    Args:
        df: DataFrame with selected columns

    Returns:
        Summary DataFrame with group counts
    """
    if df.empty:
        logger.warning("Empty DataFrame - returning empty summary")
        return pd.DataFrame(columns=[
            "dataset_name", "entity_style", "topic", "mentions",
            "clusters", "singletons", "non_singletons"
        ])

    logger.info("Creating mention groups summary...")

    # Group by the three keys
    grouped = df.groupby(["dataset_name", "entity_style", "topic"]).agg(
        mentions=("mention_id", "count"),
        clusters=("coref_chain", "nunique"),
    ).reset_index()

    # Calculate singletons and non-singletons for each group
    singleton_counts = []
    non_singleton_counts = []

    for _, row in grouped.iterrows():
        group_df = df[
            (df["dataset_name"] == row["dataset_name"]) &
            (df["entity_style"] == row["entity_style"]) &
            (df["topic"] == row["topic"])
        ]

        cluster_sizes = group_df.groupby("coref_chain").size()
        singletons = int((cluster_sizes == 1).sum())
        non_singletons = int((cluster_sizes > 1).sum())

        singleton_counts.append(singletons)
        non_singleton_counts.append(non_singletons)

    grouped["singletons"] = singleton_counts
    grouped["non_singletons"] = non_singleton_counts

    # Sort by dataset, entity_style, then mentions (descending)
    grouped = grouped.sort_values(
        ["dataset_name", "entity_style", "mentions"],
        ascending=[True, True, False]
    )

    logger.info(f"✓ Created {len(grouped)} groups")
    logger.info(f"  Breakdown:")
    logger.info(f"    Datasets: {grouped['dataset_name'].nunique()}")
    logger.info(f"    Entity styles: {grouped['entity_style'].nunique()}")
    logger.info(f"    Unique topics: {grouped['topic'].nunique()}")

    return grouped


def create_mention_string_representation(
    df: pd.DataFrame,
    data_path: str,
    special_mention_tokens: Tuple[str, str] = ("[E]", "[/E]")
) -> pd.DataFrame:
    """
    Create string representations for mentions as they will be processed by the tokenizer.

    For each mention, creates a representation with:
    - Document context: 8 sentences from the document (excluding mention's sentence)
    - Mention context: The mention's sentence with special tokens [E] and [/E]

    The tokenizer will later use max_length=128 tokens with truncation="only_first",
    which means:
    - Document context will be truncated first to fit in 128 tokens
    - Mention sentence with markers will be preserved as much as possible

    Args:
        df: DataFrame with mentions_for_pairs (must have conll_doc_key for each mention)
        data_path: Base path to raw CDCR datasets (e.g., "data/01_raw/cdcr_datasets")
        special_mention_tokens: Tuple of (start_token, end_token) to mark mentions

    Returns:
        DataFrame with columns:
        - mention_id: Unique mention identifier
        - string_representation: String to be tokenized (document_context [SEP] mention_context)
        - document_context: The 8 sentences context (for inspection)
        - mention_context: The mention sentence with markers (for inspection)
    """
    if df.empty:
        logger.warning("Empty DataFrame - returning empty representation")
        return pd.DataFrame(columns=[
            "mention_id", "string_representation", "document_context", "mention_context"
        ])

    # Print to stdout to bypass Rich logging
    import sys
    print(f"\n=== Starting create_mention_string_representation ===", file=sys.stderr)
    print(f"  Total mentions: {len(df):,}", file=sys.stderr)
    print(f"  Special tokens: {special_mention_tokens}", file=sys.stderr)

    logger.info("Creating string representations for mentions...")
    logger.info(f"  Total mentions: {len(df):,}")

    # Load all required documents
    datasets_needed = df["dataset_name"].unique()
    logger.info(f"  Loading documents from {len(datasets_needed)} datasets")

    # Dictionary to cache loaded documents
    all_documents = {}

    for dataset_name in datasets_needed:
        # Determine the dataset path
        dataset_path = os.path.join(data_path, dataset_name)

        # Print to stderr to bypass Rich
        import sys
        print(f"  Loading dataset: {dataset_name}", file=sys.stderr)

        # Load documents from parquet
        try:
            docs = _read_parquet_documents(dataset_path)
            all_documents.update(docs)
            print(f"    Loaded {len(docs)} documents from {dataset_name}", file=sys.stderr)
            logger.info(f"    Loaded {len(docs)} documents from {dataset_name}")
        except Exception as e:
            # Sanitize and print
            error_msg = str(e).replace('[', '<').replace(']', '>')
            print(f"    ERROR loading {dataset_name}: {error_msg}", file=sys.stderr)
            raise SanitizedException(f"Failed to load documents from {dataset_name}: {e}")

    # Convert documents to sentence dictionaries
    logger.info("  Converting documents to sentence dictionaries...")
    sentences_by_doc = _make_dict_of_sentences(all_documents)

    # Create representations for each mention
    logger.info("  Creating mention representations...")
    representations = []
    errors = []

    # Wrap in try-catch to prevent Rich markup errors from breaking the entire pipeline
    try:
        _process_mentions(
            df, sentences_by_doc, special_mention_tokens, representations, errors
        )
    except Exception as e:
        # Print directly to stdout to bypass Rich logging
        import sys
        print(f"\n[ERROR] Failed to process mentions:", file=sys.stderr)
        print(f"  Error type: {type(e).__name__}", file=sys.stderr)
        print(f"  Error message (sanitized): {str(e).replace('[', '(').replace(']', ')')}", file=sys.stderr)
        print(f"  Processed {len(representations)} representations before error", file=sys.stderr)
        print(f"  Accumulated {len(errors)} errors", file=sys.stderr)
        # Re-raise as SanitizedException to avoid Rich markup issues
        raise SanitizedException(f"Failed to process mentions: {e}")

    # Log errors if any
    if errors:
        logger.warning(f"  Encountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            # Sanitize error messages
            safe_error = error.replace('[', '(').replace(']', ')')
            logger.warning(f"    {safe_error}")
        if len(errors) > 10:
            logger.warning(f"    ... and {len(errors) - 10} more errors")

    logger.info(f"✓ Created {len(representations)} string representations")

    # Convert to DataFrame
    if not representations:
        return pd.DataFrame(columns=[
            "mention_id", "string_representation", "document_context", "mention_context"
        ])

    result_df = pd.DataFrame(representations)
    return result_df


def _process_mentions(df, sentences_by_doc, special_mention_tokens, representations, errors):
    """Helper function to process mentions without Rich markup issues."""
    for idx, row in df.iterrows():
        mention_id = "unknown"
        try:
            mention_id = row["mention_id"]
            conll_doc_key = row["conll_doc_key"]
            sent_id = int(row["sent_id"])
            tokens_number = row["tokens_number"]

            # Parse tokens_number if it's a string (from CSV)
            if isinstance(tokens_number, str):
                # Handle NumPy array string format: "[1 2 3]" -> [1, 2, 3]
                # Replace spaces with commas to make it valid Python list syntax
                import re
                # Remove brackets, split by whitespace, convert to ints
                tokens_str = tokens_number.strip('[]')
                tokens_number = [int(x) for x in tokens_str.split() if x]

            # Get the document's sentences
            if conll_doc_key not in sentences_by_doc:
                errors.append(f"Document key not found: {conll_doc_key} for mention {mention_id}")
                continue

            doc_sentences = sentences_by_doc[conll_doc_key]

            # Create document context (8 sentences excluding the mention's sentence)
            document_context = _create_document_context(doc_sentences, sent_id)

            # Create mention context (mention's sentence with special markers)
            mention_context = _create_mention_context(
                doc_sentences, sent_id, tokens_number, special_mention_tokens
            )

            # Combine into string representation
            # Format: "document_context (SEP) mention_context"
            # The tokenizer will process this as text_pair format
            document_context_str = " ".join(document_context).replace('\\n', '')
            mention_context_str = " ".join(mention_context).replace('\\n', '')

            # Store representation
            representations.append({
                "mention_id": mention_id,
                "string_representation": f"{document_context_str} [SEP] {mention_context_str}",
                "document_context": document_context_str,
                "mention_context": mention_context_str
            })

        except SanitizedException:
            # Already sanitized, just re-raise
            raise
        except Exception as e:
            # Sanitize and append to errors instead of raising
            error_msg = str(e).replace('[', '(').replace(']', ')')
            errors.append(f"Error processing mention {mention_id}: {error_msg}")
            continue


def _read_parquet_documents(dataset_path: str) -> dict:
    """Read documents from parquet file and convert to token list format."""
    Token = namedtuple("Token", "conll_doc_key sentence_id token_id token_str coref_chain")

    dict_documents = {}
    parquet_path = os.path.join(dataset_path, "all_documents.parquet")

    import sys
    import pyarrow.parquet as pq

    try:
        if not os.path.exists(parquet_path):
            raise SanitizedException(f"Parquet file not found: {parquet_path}")

        print(f"      Reading parquet from: {parquet_path}", file=sys.stderr)

        # Use pyarrow to read in batches to avoid OOM
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"      Total rows in parquet: {total_rows:,}", file=sys.stderr)

        # Read in batches
        batch_size = 1_000_000  # 1 million rows at a time
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            df_batch = batch.to_pandas()
            print(f"      Processing batch {batch_idx + 1} ({len(df_batch):,} rows)...", file=sys.stderr)

            # Group by document within this batch
            for doc_key, doc_group in df_batch.groupby('topic/subtopic_name/doc'):
                # Create or extend the token list for this document
                if doc_key not in dict_documents:
                    dict_documents[doc_key] = []

                for _, row in doc_group.iterrows():
                    token = Token(
                        conll_doc_key=doc_key,
                        sentence_id=row['sent_id'],
                        token_id=row['token_id'],
                        token_str=row['token'],
                        coref_chain='-'
                    )
                    dict_documents[doc_key].append(token)

        print(f"      Created {len(dict_documents)} documents", file=sys.stderr)
        return dict_documents
    except Exception as e:
        error_msg = str(e).replace('<', '(').replace('>', ')')
        print(f"      ERROR in _read_parquet_documents: {error_msg}", file=sys.stderr)
        raise SanitizedException(f"Error reading parquet documents: {e}")


def _make_dict_of_sentences(documents: dict) -> dict:
    """Convert documents to nested dict: {doc_key: {sent_id: {token_id: token_str}}}."""
    doc_dict = {}
    for doc, tokens in documents.items():
        token_dict = defaultdict(dict)
        for token in tokens:
            token_dict[token.sentence_id][token.token_id] = token.token_str
        doc_dict[doc] = token_dict

    return doc_dict


def _create_document_context(doc_sentences: dict, mention_sent_id: int) -> list:
    """
    Create document context from 8 sentences (excluding mention's sentence).

    Takes 8 sentences from positions [0,1,2,3,4,5,6,7,8], excluding the mention's sentence.
    Groups them as 4 pairs: [0,1], [2,3], [4,5], [6,7].

    Args:
        doc_sentences: Dictionary {sent_id: {token_id: token_str}}
        mention_sent_id: Sentence ID containing the mention

    Returns:
        List of tokens from 8 sentences concatenated
    """
    # Possible sentence positions
    possible_sentences = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Remove the mention's sentence
    if mention_sent_id in possible_sentences:
        possible_sentences.remove(mention_sent_id)
    else:
        # If mention is beyond sentence 8, remove the last one
        possible_sentences.pop()

    # Extract 8 sentences (now we have 8 after removing mention's sentence)
    # Group as 4 pairs
    document_context = []

    for i in range(0, min(8, len(possible_sentences)), 2):
        # Get pair of sentences
        sent_id1 = possible_sentences[i]
        sent_id2 = possible_sentences[i + 1] if i + 1 < len(possible_sentences) else None

        # Extract tokens from first sentence
        if sent_id1 in doc_sentences:
            sent1_tokens = [value for _, value in sorted(doc_sentences[sent_id1].items())]
            document_context.extend(sent1_tokens)

        # Extract tokens from second sentence
        if sent_id2 is not None and sent_id2 in doc_sentences:
            sent2_tokens = [value for _, value in sorted(doc_sentences[sent_id2].items())]
            document_context.extend(sent2_tokens)

    return document_context


def _create_mention_context(
    doc_sentences: dict,
    sent_id: int,
    tokens_number: list,
    special_mention_tokens: Tuple[str, str]
) -> list:
    """
    Create mention context with special tokens (E) and (/E) around the mention.

    Args:
        doc_sentences: Dictionary {sent_id: {token_id: token_str}}
        sent_id: Sentence ID containing the mention
        tokens_number: List of token positions that form the mention
        special_mention_tokens: Tuple of (start_token, end_token)

    Returns:
        List of tokens with special markers inserted
    """
    try:
        # Get the sentence tokens
        if sent_id not in doc_sentences:
            raise SanitizedException(f"Sentence ID {sent_id} not found in document")

        mention_context = [value for _, value in sorted(doc_sentences[sent_id].items())]

        # Determine mention boundaries
        lower_limit = min(tokens_number)
        upper_limit = max(tokens_number)

        # Insert special tokens (insert in reverse order to maintain positions)
        # Insert (/E) after the mention
        mention_context.insert(upper_limit + 1, special_mention_tokens[1])

        # Insert (E) before the mention
        mention_context.insert(lower_limit, special_mention_tokens[0])

        return mention_context
    except Exception as e:
        # Sanitize any errors that might contain brackets
        raise SanitizedException(f"Error creating mention context for sent_id {sent_id}: {str(e).replace('[', '(').replace(']', ')')}")
