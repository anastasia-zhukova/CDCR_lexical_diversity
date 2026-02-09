"""
Nodes for pairs selection pipeline.

This pipeline implements sentence-level similarity approach:
1. Extract and group sentences by topic (without [E]/[/E] markers)
2. Encode sentences with SBERT and compute pairwise similarities
3. Create mention pairs via product() of mentions in sentence pairs
4. Apply hard mining based on sentence similarities
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from itertools import product, combinations

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_sentences_by_topic(
    mentions_df: pd.DataFrame,
    data_path: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]]]:
    """
    Prepare sentences grouped by topic and map mentions to sentences.

    This function:
    1. Filters to train split only
    2. Loads raw documents and extracts full sentences (WITHOUT [E]/[/E] markers)
    3. Groups sentences by topic
    4. Maps mentions to their sentences

    Args:
        mentions_df: DataFrame with columns: mention_id, topic, conll_doc_key, sent_id, split
        data_path: Path to raw parquet documents

    Returns:
        sentences_by_topic: {topic: {sentence_id: sentence_text}}
        mentions_by_sentence: {sentence_id: {mention_ids}}
    """
    logger.info("Starting sentence preparation by topic")

    # Filter to train split only
    train_mentions = mentions_df[mentions_df["split"] == "train"].copy()
    logger.info(f"Filtered to train split: {len(train_mentions):,} mentions")

    # Initialize output structures
    sentences_by_topic = {}
    mentions_by_sentence = {}

    # Group by topic
    topics = train_mentions["topic"].unique()
    logger.info(f"Processing {len(topics)} topics")

    # Track documents we need to load
    docs_needed = {}  # {dataset: {doc_keys}}
    for _, row in train_mentions.iterrows():
        dataset = row["dataset_name"]
        doc_key = row["conll_doc_key"]
        if dataset not in docs_needed:
            docs_needed[dataset] = set()
        docs_needed[dataset].add(doc_key)

    logger.info(f"Need to load {sum(len(docs) for docs in docs_needed.values())} unique documents from {len(docs_needed)} datasets")

    # Load documents by dataset
    documents_cache = {}  # {(dataset, doc_key): document_data}

    for dataset, doc_keys in tqdm(docs_needed.items(), desc="Loading documents by dataset"):
        # Try both naming conventions
        dataset_path = Path(data_path) / dataset / "all_documents.parquet"
        if not dataset_path.exists():
            dataset_path = Path(data_path) / dataset / "documents.parquet"

        if not dataset_path.exists():
            logger.warning(f"Dataset path not found for {dataset}")
            continue

        # Read parquet file in batches - it has token-level data
        logger.info(f"  Loading {dataset}...")
        parquet_file = pq.ParquetFile(str(dataset_path))

        # The column is 'topic/subtopic_name/doc', which maps to conll_doc_key
        doc_column = 'topic/subtopic_name/doc'

        # Process in batches to avoid OOM
        batch_num = 0
        for batch in parquet_file.iter_batches(batch_size=1_000_000):
            batch_num += 1
            df_batch = batch.to_pandas()

            if doc_column not in df_batch.columns:
                logger.warning(f"Column '{doc_column}' not found in {dataset}, skipping")
                break

            # Filter to needed documents
            df_batch = df_batch[df_batch[doc_column].isin(doc_keys)]

            if len(df_batch) == 0:
                continue

            logger.info(f"    Processing batch {batch_num}: {len(df_batch):,} tokens from {df_batch[doc_column].nunique()} documents")

            # Group by document and build sentence dictionaries
            for doc_key, doc_group in df_batch.groupby(doc_column):
                cache_key = (dataset, doc_key)

                # Initialize or get existing document data
                if cache_key not in documents_cache:
                    documents_cache[cache_key] = {}  # Store as dict first, convert to list later

                # Build dict of sentences: {sent_id: {token_id: token_str}}
                sentences_dict = documents_cache[cache_key]

                for _, row in doc_group.iterrows():
                    sent_id = row['sent_id']
                    token_id = row['token_id']
                    token_str = row['token']

                    if sent_id not in sentences_dict:
                        sentences_dict[sent_id] = {}
                    sentences_dict[sent_id][token_id] = token_str

        # After all batches for this dataset, convert sentence dicts to lists
        logger.info(f"  Converting sentence dictionaries to lists for {dataset}...")
        for cache_key in documents_cache.keys():
            if cache_key[0] == dataset and isinstance(documents_cache[cache_key], dict):
                sentences_dict = documents_cache[cache_key]
                sentences_list = []
                for sent_id in sorted(sentences_dict.keys()):
                    tokens = [sentences_dict[sent_id][tid] for tid in sorted(sentences_dict[sent_id].keys())]
                    sentence_text = " ".join(tokens)
                    sentences_list.append(sentence_text)
                documents_cache[cache_key] = sentences_list

    logger.info(f"Loaded {len(documents_cache):,} documents into cache")

    # Now process mentions and extract sentences
    sentence_counter = 0

    for topic in tqdm(topics, desc="Processing topics"):
        topic_mentions = train_mentions[train_mentions["topic"] == topic]

        if topic not in sentences_by_topic:
            sentences_by_topic[topic] = {}

        # Process each mention
        for _, mention_row in topic_mentions.iterrows():
            mention_id = mention_row["mention_id"]
            dataset = mention_row["dataset_name"]
            doc_key = mention_row["conll_doc_key"]
            sent_id = mention_row["sent_id"]

            # Create unique sentence identifier
            sentence_id = f"{dataset}_{doc_key}_{sent_id}"

            # Map mention to sentence
            if sentence_id not in mentions_by_sentence:
                mentions_by_sentence[sentence_id] = set()
            mentions_by_sentence[sentence_id].add(mention_id)

            # Extract sentence if not already extracted
            if sentence_id not in sentences_by_topic[topic]:
                cache_key = (dataset, doc_key)

                if cache_key not in documents_cache:
                    logger.warning(f"Document not found in cache: {cache_key}")
                    continue

                # documents_cache[cache_key] is a list of sentence strings
                sentences_list = documents_cache[cache_key]

                # Get the specific sentence (full text, no markers)
                if sent_id < len(sentences_list):
                    sentence_text = sentences_list[sent_id]
                    sentences_by_topic[topic][sentence_id] = sentence_text
                    sentence_counter += 1
                else:
                    logger.warning(f"sent_id {sent_id} out of range for {cache_key}")

    logger.info(f"Extracted {sentence_counter:,} unique sentences")
    logger.info(f"Mapped {len(mentions_by_sentence):,} sentences to mentions")
    logger.info(f"Topics: {len(sentences_by_topic)}")

    # Log statistics per topic
    for topic, sentences in sentences_by_topic.items():
        mentions_count = sum(
            len(mentions_by_sentence.get(sent_id, set()))
            for sent_id in sentences.keys()
        )
        logger.info(f"  Topic '{topic[:50]}...': {len(sentences)} sentences, {mentions_count} mentions")

    return sentences_by_topic, mentions_by_sentence


def encode_sentences_by_topic(
    sentences_by_topic: Dict[str, Dict[str, str]],
    mentions_by_sentence: Dict[str, Set[str]],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_batch_size: int = 1000
) -> pd.DataFrame:
    """
    Encode sentences (not mentions) and compute pairwise similarities.

    For each topic:
    1. Encode all sentences with SBERT
    2. Compute pairwise cosine similarity in batches (lower triangle only)
    3. Filter to sentence pairs where BOTH sentences have mentions
    4. Filter to sentence pairs where total mentions >= 2
    5. Filter out exact duplicates (similarity = 1.0)

    Args:
        sentences_by_topic: {topic: {sentence_id: sentence_text}}
        mentions_by_sentence: {sentence_id: {mention_ids}}
        model_name: SBERT model to use for encoding
        similarity_batch_size: Number of sentences to process per batch for similarity computation
                               (prevents OOM on large topics)

    Returns:
        DataFrame with columns: sentence_id1, sentence_id2, similarity, topic
    """
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    all_pairs = []

    for topic, sentences in tqdm(sentences_by_topic.items(), desc="Encoding sentences by topic"):
        # Skip topics with too few sentences
        if len(sentences) < 2:
            logger.warning(f"Topic '{topic[:50]}...' has only {len(sentences)} sentence(s), skipping")
            continue

        # Prepare sentences for encoding
        sentence_ids = list(sentences.keys())
        sentence_texts = [sentences[sid] for sid in sentence_ids]

        # Encode all sentences
        logger.info(f"Encoding {len(sentence_texts)} sentences for topic '{topic[:50]}...'")
        embeddings = model.encode(
            sentence_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Compute pairwise cosine similarities in batches to avoid OOM
        n_sentences = len(sentence_ids)
        pair_count = 0
        filtered_duplicate = 0
        filtered_no_mentions = 0

        logger.info(f"Computing similarities in batches of {similarity_batch_size}...")

        # Process in batches to avoid loading full N×N similarity matrix
        for batch_start in range(0, n_sentences, similarity_batch_size):
            batch_end = min(batch_start + similarity_batch_size, n_sentences)
            batch_size_actual = batch_end - batch_start

            # Get embeddings for this batch
            batch_embeddings = embeddings[batch_start:batch_end]

            # Compute similarity between this batch and ALL sentences
            # Shape: (batch_size, n_sentences)
            batch_scores = util.cos_sim(batch_embeddings, embeddings)

            logger.info(f"  Processing batch {batch_start}-{batch_end-1} ({batch_size_actual} sentences)")

            # Extract pairs from this batch
            # For each sentence i in the batch, compare with sentences j where j > i
            for i_batch in range(batch_size_actual):
                i_global = batch_start + i_batch

                # Only look at j > i_global (lower triangle, avoid self-similarity)
                for j_global in range(i_global + 1, n_sentences):
                    sent_id1 = sentence_ids[i_global]
                    sent_id2 = sentence_ids[j_global]
                    similarity = batch_scores[i_batch][j_global].item()

                    # Filter: exact duplicates
                    if similarity >= 0.9999:  # Essentially 1.0
                        filtered_duplicate += 1
                        continue

                    # Filter: both sentences must have mentions
                    mentions1 = mentions_by_sentence.get(sent_id1, set())
                    mentions2 = mentions_by_sentence.get(sent_id2, set())

                    if len(mentions1) == 0 or len(mentions2) == 0:
                        filtered_no_mentions += 1
                        continue

                    # Filter: total mentions in pair must be >= 2
                    total_mentions = len(mentions1.union(mentions2))
                    if total_mentions < 2:
                        filtered_no_mentions += 1
                        continue

                    # Keep this pair
                    all_pairs.append({
                        "sentence_id1": sent_id1,
                        "sentence_id2": sent_id2,
                        "similarity": similarity,
                        "topic": topic
                    })
                    pair_count += 1

            # Free memory from this batch
            del batch_scores

        logger.info(f"  Topic '{topic[:50]}...': {pair_count:,} sentence pairs kept")
        logger.info(f"    Filtered duplicates: {filtered_duplicate:,}")
        logger.info(f"    Filtered no/few mentions: {filtered_no_mentions:,}")

    df = pd.DataFrame(all_pairs)
    logger.info(f"Total sentence pairs across all topics: {len(df):,}")

    return df


def create_mention_pairs_from_sentences(
    sentence_pairs_df: pd.DataFrame,
    mentions_by_sentence: Dict[str, Set[str]],
    mentions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each sentence pair, create all mention combinations.

    Process:
    1. For each (sent1, sent2) with similarity:
       - Get mentions from sent1
       - Get mentions from sent2
       - Create product(mentions_sent1, mentions_sent2)
       - Assign sentence similarity to all mention pairs
       - Label: is_positive = (coref_chain1 == coref_chain2)

    Args:
        sentence_pairs_df: DataFrame with sentence_id1, sentence_id2, similarity, topic
        mentions_by_sentence: {sentence_id: {mention_ids}}
        mentions_df: DataFrame with mention_id, coref_chain columns

    Returns:
        DataFrame: mention_id1, mention_id2, sentence_similarity, is_positive, topic
    """
    logger.info("Creating mention pairs from sentence pairs")

    # Create mention lookup for coref chains
    mention_to_coref = mentions_df.set_index("mention_id")["coref_chain"].to_dict()

    all_mention_pairs = []

    for _, row in tqdm(sentence_pairs_df.iterrows(), total=len(sentence_pairs_df), desc="Creating mention pairs"):
        sent_id1 = row["sentence_id1"]
        sent_id2 = row["sentence_id2"]
        similarity = row["similarity"]
        topic = row["topic"]

        # Get mentions from both sentences
        mentions1 = mentions_by_sentence.get(sent_id1, set())
        mentions2 = mentions_by_sentence.get(sent_id2, set())

        # Create all combinations (Cartesian product)
        for m1, m2 in product(mentions1, mentions2):
            # Get coref chains
            coref1 = mention_to_coref.get(m1)
            coref2 = mention_to_coref.get(m2)

            if coref1 is None or coref2 is None:
                continue

            # Determine if positive or negative
            is_positive = (coref1 == coref2)

            all_mention_pairs.append({
                "mention_id1": m1,
                "mention_id2": m2,
                "sentence_similarity": similarity,
                "is_positive": is_positive,
                "topic": topic
            })

    df = pd.DataFrame(all_mention_pairs)

    logger.info(f"Created {len(df):,} mention pairs from sentence pairs")
    logger.info(f"  Positive pairs: {df['is_positive'].sum():,} ({df['is_positive'].sum()/len(df)*100:.1f}%)")
    logger.info(f"  Negative pairs: {(~df['is_positive']).sum():,} ({(~df['is_positive']).sum()/len(df)*100:.1f}%)")

    # Add dataset and entity_style information from mentions_df
    logger.info("Adding dataset and entity_style information to pairs")

    # Create lookup for dataset_name and entity_style
    mention_info = mentions_df.set_index("mention_id")[["dataset_name", "entity_style"]].to_dict("index")

    # Add dataset_name and entity_style for both mentions
    df["dataset1"] = df["mention_id1"].map(lambda x: mention_info.get(x, {}).get("dataset_name", "unknown"))
    df["entity_style1"] = df["mention_id1"].map(lambda x: mention_info.get(x, {}).get("entity_style", "unknown"))
    df["dataset2"] = df["mention_id2"].map(lambda x: mention_info.get(x, {}).get("dataset_name", "unknown"))
    df["entity_style2"] = df["mention_id2"].map(lambda x: mention_info.get(x, {}).get("entity_style", "unknown"))

    # For summary, use dataset1 (they should be same topic, so same dataset)
    df["dataset"] = df["dataset1"]
    df["entity_style"] = df["entity_style1"]

    # Log statistics by dataset
    logger.info("\n" + "="*80)
    logger.info("MENTION PAIRS SUMMARY BY DATASET")
    logger.info("="*80)

    dataset_stats = df.groupby("dataset").agg({
        "is_positive": ["sum", "count", lambda x: (~x).sum()]
    }).reset_index()
    dataset_stats.columns = ["dataset", "positive_pairs", "total_pairs", "negative_pairs"]
    dataset_stats["positive_pct"] = (dataset_stats["positive_pairs"] / dataset_stats["total_pairs"] * 100).round(1)
    dataset_stats = dataset_stats.sort_values("total_pairs", ascending=False)

    for _, row in dataset_stats.iterrows():
        logger.info(f"\nDataset: {row['dataset']}")
        logger.info(f"  Total pairs: {int(row['total_pairs']):,}")
        logger.info(f"  Positive pairs: {int(row['positive_pairs']):,} ({row['positive_pct']:.1f}%)")
        logger.info(f"  Negative pairs: {int(row['negative_pairs']):,} ({100-row['positive_pct']:.1f}%)")

    # Log statistics by entity_style
    logger.info("\n" + "="*80)
    logger.info("MENTION PAIRS SUMMARY BY ENTITY_STYLE")
    logger.info("="*80)

    style_stats = df.groupby("entity_style").agg({
        "is_positive": ["sum", "count", lambda x: (~x).sum()]
    }).reset_index()
    style_stats.columns = ["entity_style", "positive_pairs", "total_pairs", "negative_pairs"]
    style_stats["positive_pct"] = (style_stats["positive_pairs"] / style_stats["total_pairs"] * 100).round(1)
    style_stats = style_stats.sort_values("total_pairs", ascending=False)

    for _, row in style_stats.iterrows():
        logger.info(f"\nEntity Style: {row['entity_style']}")
        logger.info(f"  Total pairs: {int(row['total_pairs']):,}")
        logger.info(f"  Positive pairs: {int(row['positive_pairs']):,} ({row['positive_pct']:.1f}%)")
        logger.info(f"  Negative pairs: {int(row['negative_pairs']):,} ({100-row['positive_pct']:.1f}%)")

    # Log statistics by dataset AND entity_style
    logger.info("\n" + "="*80)
    logger.info("MENTION PAIRS SUMMARY BY DATASET AND ENTITY_STYLE")
    logger.info("="*80)

    combined_stats = df.groupby(["dataset", "entity_style"]).agg({
        "is_positive": ["sum", "count", lambda x: (~x).sum()]
    }).reset_index()
    combined_stats.columns = ["dataset", "entity_style", "positive_pairs", "total_pairs", "negative_pairs"]
    combined_stats["positive_pct"] = (combined_stats["positive_pairs"] / combined_stats["total_pairs"] * 100).round(1)
    combined_stats = combined_stats.sort_values(["dataset", "total_pairs"], ascending=[True, False])

    current_dataset = None
    for _, row in combined_stats.iterrows():
        if current_dataset != row['dataset']:
            logger.info(f"\n{row['dataset']}:")
            current_dataset = row['dataset']
        logger.info(f"  {row['entity_style']}: {int(row['total_pairs']):,} pairs ({int(row['positive_pairs']):,} pos, {int(row['negative_pairs']):,} neg)")

    logger.info("\n" + "="*80)

    return df


def select_hard_pairs_sentence_based(
    mention_pairs_df: pd.DataFrame,
    additional_sampling: bool = False
) -> pd.DataFrame:
    """
    Select hard pairs based on SENTENCE similarities.

    For each topic:
    1. Compute topic sentence similarity median
    2. Hard positives: is_positive=1 AND sentence_sim < topic_median
    3. Hard negatives: is_positive=0 AND sentence_sim > Q3(positive_sims)

    Args:
        mention_pairs_df: DataFrame with sentence_similarity, is_positive, topic
        additional_sampling: Whether to apply sampling (reserved for future use)

    Returns:
        DataFrame: Selected pairs with labels
    """
    logger.info("Selecting hard pairs based on sentence similarities")

    selected_pairs = []

    topics = mention_pairs_df["topic"].unique()

    for topic in tqdm(topics, desc="Selecting hard pairs by topic"):
        topic_pairs = mention_pairs_df[mention_pairs_df["topic"] == topic]

        # Compute topic-level statistics
        topic_sim_median = topic_pairs["sentence_similarity"].median()

        # Separate positive and negative pairs
        positive_pairs = topic_pairs[topic_pairs["is_positive"]]
        negative_pairs = topic_pairs[~topic_pairs["is_positive"]]

        if len(positive_pairs) == 0 or len(negative_pairs) == 0:
            logger.warning(f"Topic '{topic[:50]}...' has no positive or negative pairs, skipping")
            continue

        # Compute Q3 for positive similarities
        q3_positive = positive_pairs["sentence_similarity"].quantile(0.75)

        # Hard positives: same coref, LOW sentence similarity
        hard_positives = positive_pairs[
            positive_pairs["sentence_similarity"] < topic_sim_median
        ]

        # Hard negatives: different coref, HIGH sentence similarity
        hard_negatives = negative_pairs[
            negative_pairs["sentence_similarity"] > q3_positive
        ]

        logger.info(f"Topic '{topic[:50]}...':")
        logger.info(f"  Topic sentence similarity median: {topic_sim_median:.3f}")
        logger.info(f"  Q3 positive similarities: {q3_positive:.3f}")
        logger.info(f"  Hard positives: {len(hard_positives):,} / {len(positive_pairs):,} ({len(hard_positives)/len(positive_pairs)*100:.1f}%)")
        logger.info(f"  Hard negatives: {len(hard_negatives):,} / {len(negative_pairs):,} ({len(hard_negatives)/len(negative_pairs)*100:.1f}%)")

        # Optional sampling (disabled for now to see full distribution)
        if additional_sampling:
            logger.warning("Additional sampling is not yet implemented")
            # TODO: Implement sampling strategy after analyzing similarity distributions

        # Combine
        topic_selected = pd.concat([hard_positives, hard_negatives], ignore_index=True)
        selected_pairs.append(topic_selected)

    df = pd.DataFrame(pd.concat(selected_pairs, ignore_index=True))

    logger.info(f"\nFinal selected pairs: {len(df):,}")
    logger.info(f"  Positive: {df['is_positive'].sum():,} ({df['is_positive'].sum()/len(df)*100:.1f}%)")
    logger.info(f"  Negative: {(~df['is_positive']).sum():,} ({(~df['is_positive']).sum()/len(df)*100:.1f}%)")

    return df
