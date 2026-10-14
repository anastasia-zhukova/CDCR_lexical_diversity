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
import gc
import psutil
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set
from itertools import product, combinations

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

logger = logging.getLogger(__name__)


def log_memory_usage(prefix: str = ""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    mem_percent = process.memory_percent()

    vm = psutil.virtual_memory()
    system_used_mb = vm.used / 1024 / 1024
    system_total_mb = vm.total / 1024 / 1024
    system_available_mb = vm.available / 1024 / 1024
    system_percent = vm.percent

    logger.info(f"[MEMORY {prefix}] Process: {mem_mb:.1f} MB ({mem_percent:.1f}%)")
    logger.info(f"[MEMORY {prefix}] System: {system_used_mb:.1f}/{system_total_mb:.1f} MB ({system_percent:.1f}% used, {system_available_mb:.1f} MB available)")

    return mem_mb, system_percent


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
    log_memory_usage("START prepare_sentences_by_topic")

    # Filter to train split only
    train_mentions = mentions_df[mentions_df["split"] == "train"].copy()
    logger.info(f"Filtered to train split: {len(train_mentions):,} mentions")
    log_memory_usage("AFTER filtering train split")

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

    # Load documents by dataset - REDUCED BATCH SIZE for memory control
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
        log_memory_usage(f"BEFORE loading {dataset}")
        parquet_file = pq.ParquetFile(str(dataset_path))

        # The column is 'topic/subtopic_name/doc', which maps to conll_doc_key
        doc_column = 'topic/subtopic_name/doc'

        # Process in SMALLER batches to avoid OOM (reduced from 1M to 500K)
        batch_num = 0
        for batch in parquet_file.iter_batches(batch_size=500_000):
            batch_num += 1
            df_batch = batch.to_pandas()

            if doc_column not in df_batch.columns:
                logger.warning(f"Column '{doc_column}' not found in {dataset}, skipping")
                break

            # Filter to needed documents
            df_batch = df_batch[df_batch[doc_column].isin(doc_keys)]

            if len(df_batch) == 0:
                del df_batch
                gc.collect()
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

            # Clean up batch immediately
            del df_batch
            gc.collect()

            # Log memory every 3 batches
            if batch_num % 3 == 0:
                log_memory_usage(f"AFTER batch {batch_num} of {dataset}")

        # After all batches for this dataset, convert sentence dicts to lists
        logger.info(f"  Converting sentence dictionaries to lists for {dataset}...")
        for cache_key in list(documents_cache.keys()):
            if cache_key[0] == dataset and isinstance(documents_cache[cache_key], dict):
                sentences_dict = documents_cache[cache_key]
                sentences_list = []
                for sent_id in sorted(sentences_dict.keys()):
                    tokens = [sentences_dict[sent_id][tid] for tid in sorted(sentences_dict[sent_id].keys())]
                    sentence_text = " ".join(tokens)
                    sentences_list.append(sentence_text)
                documents_cache[cache_key] = sentences_list

        # Force garbage collection after each dataset
        gc.collect()
        log_memory_usage(f"AFTER processing {dataset}")

    logger.info(f"Loaded {len(documents_cache):,} documents into cache")
    log_memory_usage("AFTER loading all documents")

    # Now process mentions and extract sentences
    # Process topics ONE AT A TIME and clear cache after each to reduce memory
    sentence_counter = 0

    for topic_idx, topic in enumerate(tqdm(topics, desc="Processing topics"), 1):
        logger.info(f"Processing topic {topic_idx}/{len(topics)}: '{topic[:50]}...'")
        log_memory_usage(f"BEFORE topic {topic_idx}")

        topic_mentions = train_mentions[train_mentions["topic"] == topic]

        if topic not in sentences_by_topic:
            sentences_by_topic[topic] = {}

        # Track which documents this topic needs
        topic_docs_needed = set()
        for _, mention_row in topic_mentions.iterrows():
            dataset = mention_row["dataset_name"]
            doc_key = mention_row["conll_doc_key"]
            topic_docs_needed.add((dataset, doc_key))

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

        logger.info(f"  Topic '{topic[:50]}...': extracted {len(sentences_by_topic[topic])} sentences")
        log_memory_usage(f"AFTER topic {topic_idx}")

        # AGGRESSIVE CLEANUP: Remove documents from cache that are no longer needed
        # Check which documents are needed by remaining topics
        remaining_topics = topics[topic_idx:]  # Topics not yet processed
        remaining_docs_needed = set()
        for remaining_topic in remaining_topics:
            remaining_topic_mentions = train_mentions[train_mentions["topic"] == remaining_topic]
            for _, mention_row in remaining_topic_mentions.iterrows():
                dataset = mention_row["dataset_name"]
                doc_key = mention_row["conll_doc_key"]
                remaining_docs_needed.add((dataset, doc_key))

        # Remove documents that are no longer needed
        docs_to_remove = []
        for cache_key in documents_cache.keys():
            if cache_key not in remaining_docs_needed:
                docs_to_remove.append(cache_key)

        if docs_to_remove:
            logger.info(f"  Clearing {len(docs_to_remove)} documents from cache (no longer needed)")
            for cache_key in docs_to_remove:
                del documents_cache[cache_key]
            gc.collect()
            log_memory_usage(f"AFTER clearing cache for topic {topic_idx}")

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

    # Final cleanup
    del documents_cache
    gc.collect()
    log_memory_usage("END prepare_sentences_by_topic")

    return sentences_by_topic, mentions_by_sentence


def encode_sentences_by_topic(
    sentences_by_topic: Dict[str, Dict[str, str]],
    mentions_by_sentence: Dict[str, Set[str]],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_batch_size: int = 250
) -> pd.DataFrame:
    """
    Encode sentences (not mentions) and compute pairwise similarities.

    For each topic:
    1. Encode all sentences with SBERT
    2. Compute pairwise cosine similarity in batches (lower triangle only)
    3. Filter to sentence pairs where BOTH sentences have mentions
    4. Filter to sentence pairs where total mentions >= 2
    5. Filter out exact duplicates (similarity = 1.0)
    6. Write topic pairs to disk immediately (prevents OOM)

    Args:
        sentences_by_topic: {topic: {sentence_id: sentence_text}}
        mentions_by_sentence: {sentence_id: {mention_ids}}
        model_name: SBERT model to use for encoding
        similarity_batch_size: Number of sentences to process per batch for similarity computation
                               (prevents OOM on large topics). Default: 250

    Returns:
        Dict of DataFrames (for PartitionedDataset) with columns: sentence_id1, sentence_id2, similarity, topic
    """
    logger.info(f"Loading SBERT model: {model_name}")
    log_memory_usage("BEFORE loading SBERT model")
    model = SentenceTransformer(model_name)
    log_memory_usage("AFTER loading SBERT model")

    # Create temporary directory for storing pairs on disk
    temp_dir = tempfile.mkdtemp(prefix="pairs_selection_")
    logger.info(f"Created temporary directory for pairs: {temp_dir}")
    temp_files = []

    total_topics = len(sentences_by_topic)

    for topic_idx, (topic, sentences) in enumerate(tqdm(sentences_by_topic.items(), desc="Encoding sentences by topic"), 1):
        logger.info(f"Processing topic {topic_idx}/{total_topics}: '{topic[:50]}...'")
        log_memory_usage(f"START topic {topic_idx}")

        # Store pairs in chunks and write to disk periodically during similarity computation
        topic_pairs_chunk = []
        topic_chunk_num = 0
        PAIRS_CHUNK_SIZE = 100000  # Write every 100K pairs within a topic
        # Skip topics with too few sentences
        if len(sentences) < 2:
            logger.warning(f"Topic '{topic[:50]}...' has only {len(sentences)} sentence(s), skipping")
            continue

        # Prepare sentences for encoding
        sentence_ids = list(sentences.keys())
        sentence_texts = [sentences[sid] for sid in sentence_ids]

        # Encode all sentences
        logger.info(f"Encoding {len(sentence_texts)} sentences for topic '{topic[:50]}...'")
        log_memory_usage(f"BEFORE encoding {len(sentence_texts)} sentences")

        embeddings = model.encode(
            sentence_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=32  # Reduce batch size during encoding to control memory
        )

        # Move embeddings to CPU to reduce memory pressure
        embeddings = embeddings.cpu()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_memory_usage(f"AFTER encoding (embeddings shape: {embeddings.shape})")

        # Free the sentence texts immediately
        del sentence_texts
        gc.collect()

        # Compute pairwise cosine similarities in batches to avoid OOM
        n_sentences = len(sentence_ids)
        pair_count = 0
        filtered_duplicate = 0
        filtered_no_mentions = 0

        logger.info(f"Computing similarities in batches of {similarity_batch_size} (n_sentences={n_sentences})...")
        log_memory_usage(f"BEFORE similarity computation")

        # Process in batches to avoid loading full N×N similarity matrix
        batch_num = 0
        for batch_start in range(0, n_sentences, similarity_batch_size):
            batch_num += 1
            batch_end = min(batch_start + similarity_batch_size, n_sentences)
            batch_size_actual = batch_end - batch_start

            # Get embeddings for this batch
            batch_embeddings = embeddings[batch_start:batch_end]

            # Compute similarity between this batch and ALL sentences
            # Shape: (batch_size, n_sentences)
            batch_scores = util.cos_sim(batch_embeddings, embeddings)

            logger.info(f"  Batch {batch_num} [{batch_start}-{batch_end-1}] ({batch_size_actual} sentences)")

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

                    # Keep this pair in chunk
                    topic_pairs_chunk.append({
                        "sentence_id1": sent_id1,
                        "sentence_id2": sent_id2,
                        "similarity": similarity,
                        "topic": topic
                    })
                    pair_count += 1

            # Free memory from this batch
            del batch_scores, batch_embeddings
            gc.collect()

            # Write chunk to disk if it reaches PAIRS_CHUNK_SIZE (prevents OOM within topic)
            if len(topic_pairs_chunk) >= PAIRS_CHUNK_SIZE:
                topic_chunk_num += 1
                chunk_df = pd.DataFrame(topic_pairs_chunk)
                temp_file = os.path.join(temp_dir, f"topic_{topic_idx:04d}_chunk_{topic_chunk_num:04d}.parquet")
                chunk_df.to_parquet(temp_file, index=False)
                temp_files.append(temp_file)
                logger.info(f"    Wrote chunk {topic_chunk_num} ({len(chunk_df):,} pairs) to disk during topic processing")

                # Clear chunk
                del chunk_df
                topic_pairs_chunk = []
                gc.collect()
                log_memory_usage(f"AFTER writing topic {topic_idx} chunk {topic_chunk_num}")

            # Log memory every 5 batches
            if batch_num % 5 == 0:
                log_memory_usage(f"AFTER similarity batch {batch_num}")

        # Clean up embeddings after processing topic
        del embeddings, sentence_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  Topic '{topic[:50]}...': {pair_count:,} sentence pairs kept")
        logger.info(f"    Filtered duplicates: {filtered_duplicate:,}")
        logger.info(f"    Filtered no/few mentions: {filtered_no_mentions:,}")

        # WRITE ANY REMAINING PAIRS IN FINAL CHUNK to avoid OOM
        if topic_pairs_chunk:
            topic_chunk_num += 1
            chunk_df = pd.DataFrame(topic_pairs_chunk)
            temp_file = os.path.join(temp_dir, f"topic_{topic_idx:04d}_chunk_{topic_chunk_num:04d}.parquet")
            chunk_df.to_parquet(temp_file, index=False)
            temp_files.append(temp_file)
            logger.info(f"  Wrote final chunk {topic_chunk_num} ({len(chunk_df):,} pairs) to disk")

            # Free memory immediately
            del chunk_df, topic_pairs_chunk
            gc.collect()
            log_memory_usage(f"AFTER writing topic {topic_idx} final chunk to disk")
        else:
            if pair_count == 0:
                logger.warning(f"  No pairs generated for topic {topic_idx}")

        log_memory_usage(f"END topic {topic_idx}")

    # Return dict of DataFrames for PartitionedDataset (avoids OOM from concatenation)
    logger.info(f"Preparing {len(temp_files)} parquet chunks as dict for PartitionedDataset...")
    log_memory_usage("BEFORE reading parquet files")

    if not temp_files:
        logger.warning("No pairs generated across all topics!")
        return {"empty": pd.DataFrame(columns=["sentence_id1", "sentence_id2", "similarity", "topic"])}

    # Read chunks and return as dict (avoids concatenating 113M+ rows)
    partitions = {}
    total_pairs = 0
    for i, temp_file in enumerate(temp_files):
        df_chunk = pd.read_parquet(temp_file)
        partition_name = f"chunk_{i:05d}"
        partitions[partition_name] = df_chunk
        total_pairs += len(df_chunk)
        logger.info(f"  Loaded chunk {i+1}/{len(temp_files)}: {len(df_chunk):,} pairs")

    logger.info(f"Total sentence pairs across all topics: {total_pairs:,}")
    logger.info(f"Returning {len(partitions)} partitions (avoids OOM from concatenation)")
    log_memory_usage("AFTER loading all partitions")

    # Clean up temporary files
    logger.info(f"Cleaning up temporary directory: {temp_dir}")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to remove {temp_file}: {e}")
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to remove {temp_dir}: {e}")

    log_memory_usage("END encode_sentences_by_topic")

    return partitions


def create_mention_pairs_from_sentences(
    sentence_pairs_dict: Dict[str, pd.DataFrame],
    mentions_by_sentence: Dict[str, Set[str]],
    mentions_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    For each sentence pair, create all mention combinations.

    Process:
    1. For each (sent1, sent2) with similarity:
       - Get mentions from sent1
       - Get mentions from sent2
       - Create product(mentions_sent1, mentions_sent2)
       - Assign sentence similarity to all mention pairs
       - Label: is_positive = (coref_chain1 == coref_chain2)
    2. Write to disk in chunks to avoid OOM

    Args:
        sentence_pairs_dict: Dict of DataFrames (from PartitionedDataset) with sentence_id1, sentence_id2, similarity, topic
        mentions_by_sentence: {sentence_id: {mention_ids}}
        mentions_df: DataFrame with mention_id, coref_chain columns

    Returns:
        Dict of DataFrames: mention_id1, mention_id2, sentence_similarity, is_positive, topic
    """
    logger.info("Creating mention pairs from sentence pairs (PartitionedDataset)")
    log_memory_usage("START create_mention_pairs_from_sentences")

    # Drop duplicates before creating lookups (mentions_df may have duplicate mention_ids)
    mentions_unique = mentions_df.drop_duplicates(subset="mention_id")
    logger.info(f"Mentions: {len(mentions_df):,} total, {len(mentions_unique):,} unique mention_ids")

    # Create mention lookup for coref chains
    mention_to_coref = mentions_unique.set_index("mention_id")["coref_chain"].to_dict()

    # Create lookup for dataset_name and entity_style
    mention_info = mentions_unique.set_index("mention_id")[["dataset_name", "entity_style"]].to_dict("index")

    # Process each partition separately to avoid loading all 113M rows
    # Write output partitions to temp files immediately to avoid OOM
    temp_output_dir = tempfile.mkdtemp(prefix="mention_pairs_output_")
    logger.info(f"Created temp output directory: {temp_output_dir}")

    output_partition_files = {}
    partition_num = 0

    # Get partition keys (PartitionedDataset returns dict-like object)
    partition_keys = list(sentence_pairs_dict.keys())
    logger.info(f"Processing {len(partition_keys)} partitions")

    for partition_key in tqdm(partition_keys, desc="Processing sentence pair partitions"):
        # Load the partition (PartitionedDataset may return callable/loader)
        partition_data = sentence_pairs_dict[partition_key]

        # Check if it's callable and needs to be called to load
        if callable(partition_data):
            sentence_pairs_df = partition_data()
        else:
            sentence_pairs_df = partition_data

        logger.info(f"Processing partition {partition_key}: {len(sentence_pairs_df):,} sentence pairs")
        log_memory_usage(f"START partition {partition_key}")

        # Process this partition
        mention_pairs_list = []

        for _, row in sentence_pairs_df.iterrows():
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

                mention_pairs_list.append({
                    "mention_id1": m1,
                    "mention_id2": m2,
                    "sentence_similarity": similarity,
                    "is_positive": is_positive,
                    "topic": topic
                })

        # Convert to DataFrame for this partition
        if mention_pairs_list:
            partition_df = pd.DataFrame(mention_pairs_list)

            # Add dataset and entity_style information
            partition_df["dataset1"] = partition_df["mention_id1"].map(lambda x: mention_info.get(x, {}).get("dataset_name", "unknown"))
            partition_df["entity_style1"] = partition_df["mention_id1"].map(lambda x: mention_info.get(x, {}).get("entity_style", "unknown"))
            partition_df["dataset2"] = partition_df["mention_id2"].map(lambda x: mention_info.get(x, {}).get("dataset_name", "unknown"))
            partition_df["entity_style2"] = partition_df["mention_id2"].map(lambda x: mention_info.get(x, {}).get("entity_style", "unknown"))
            partition_df["dataset"] = partition_df["dataset1"]
            partition_df["entity_style"] = partition_df["entity_style1"]

            # Write to temp file immediately to avoid OOM
            partition_num += 1
            output_partition_name = f"partition_{partition_num:05d}"
            output_file = os.path.join(temp_output_dir, f"{output_partition_name}.parquet")
            partition_df.to_parquet(output_file, index=False)
            output_partition_files[output_partition_name] = output_file

            logger.info(f"  Created {len(partition_df):,} mention pairs from partition {partition_key}")
            logger.info(f"  Wrote to temp file: {output_partition_name}.parquet")

            # Clean up immediately
            del mention_pairs_list, partition_df, sentence_pairs_df
            gc.collect()
            log_memory_usage(f"AFTER processing partition {partition_key}")
        else:
            logger.warning(f"  No mention pairs created from partition {partition_key}")
            del mention_pairs_list, sentence_pairs_df
            gc.collect()

    # Return lazy loaders that read from temp files (avoids loading all 403 DataFrames in memory)
    if not output_partition_files:
        logger.warning("No mention pairs generated!")
        return {"empty": pd.DataFrame(columns=["mention_id1", "mention_id2", "sentence_similarity", "is_positive", "topic"])}

    logger.info(f"\nCreated mention pairs from {len(output_partition_files)} partitions")
    logger.info("  (Statistics calculation skipped to avoid loading all partitions)")

    # Create lazy loaders that mimic PartitionedDataset behavior
    def create_loader(filepath):
        """Create a lazy loader function for a parquet file."""
        def load():
            return pd.read_parquet(filepath)
        return load

    output_loaders = {}
    for partition_name, filepath in output_partition_files.items():
        output_loaders[partition_name] = create_loader(filepath)

    logger.info("\n" + "="*80)
    logger.info("Mention pairs stored as lazy loaders (avoids OOM from loading all partitions)")
    logger.info(f"Total partitions: {len(output_loaders)}")
    logger.info(f"Temp directory: {temp_output_dir}")
    logger.info("Note: Temp files will be cleaned up after Kedro saves to final location")
    logger.info("="*80)

    log_memory_usage("END create_mention_pairs_from_sentences")

    return output_loaders


def filter_pairs_by_similarity_quantiles(
    mention_pairs_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Filter mention pairs based on dataset-specific similarity quantiles.

    For each dataset:
    - Positive pairs (is_positive=True): Keep pairs with similarity < 65th percentile
    - Negative pairs (is_positive=False): Keep pairs with similarity > 96th percentile

    Args:
        mention_pairs_dict: Dict of DataFrames (from PartitionedDataset) with columns:
                           mention_id1, mention_id2, sentence_similarity, is_positive, dataset

    Returns:
        Dict of filtered DataFrames with columns: dataset, mention1_id, mention2_id, is_positive, similarity
    """
    logger.info("Filtering pairs by similarity quantiles (per dataset)")
    log_memory_usage("START filter_pairs_by_similarity_quantiles")

    # Step 1: Compute quantiles per (dataset, is_positive)
    logger.info("Step 1: Computing similarity quantiles per dataset and is_positive")
    partition_keys = list(mention_pairs_dict.keys())

    # Accumulate similarities to compute quantiles
    group_similarities = {}  # {(dataset, is_positive): [similarities]}

    for partition_key in tqdm(partition_keys, desc="Accumulating similarities for quantiles"):
        partition_data = mention_pairs_dict[partition_key]
        if callable(partition_data):
            df = partition_data()
        else:
            df = partition_data

        for (dataset, is_positive), group in df.groupby(['dataset', 'is_positive']):
            key = (dataset, is_positive)
            if key not in group_similarities:
                group_similarities[key] = []
            group_similarities[key].extend(group['sentence_similarity'].tolist())

        del df
        gc.collect()

    # Compute quantile thresholds
    quantile_thresholds = {}  # {(dataset, is_positive): threshold}
    logger.info("\nComputed quantile thresholds:")
    logger.info("-" * 80)

    for (dataset, is_positive), similarities in group_similarities.items():
        similarities_array = np.array(similarities)

        if is_positive:
            # For positive pairs: 65th percentile
            threshold = np.percentile(similarities_array, 65)
            condition = "< 65th percentile"
        else:
            # For negative pairs: 96th percentile
            threshold = np.percentile(similarities_array, 96)
            condition = "> 96th percentile"

        quantile_thresholds[(dataset, is_positive)] = threshold
        label = "Positive" if is_positive else "Negative"
        logger.info(f"  {dataset} ({label}): threshold={threshold:.4f} ({condition})")

        del similarities_array

    # Free memory
    del group_similarities
    gc.collect()
    log_memory_usage("AFTER computing quantiles")

    # Step 2: Filter pairs based on thresholds
    logger.info("\nStep 2: Filtering pairs based on thresholds")
    filtered_partitions = {}
    partition_num = 0

    # Track counts for summary
    summary_counts = {}  # {(dataset, is_positive): count}

    for partition_key in tqdm(partition_keys, desc="Filtering partitions"):
        partition_data = mention_pairs_dict[partition_key]
        if callable(partition_data):
            df = partition_data()
        else:
            df = partition_data

        # Apply filtering based on thresholds
        filtered_rows = []

        for _, row in df.iterrows():
            dataset = row['dataset']
            is_positive = row['is_positive']
            similarity = row['sentence_similarity']

            threshold = quantile_thresholds.get((dataset, is_positive))
            if threshold is None:
                continue

            # Apply filter condition
            keep = False
            if is_positive:
                # Positive pairs: keep if similarity < 65th percentile
                keep = similarity < threshold
            else:
                # Negative pairs: keep if similarity > 96th percentile
                keep = similarity > threshold

            if keep:
                filtered_rows.append({
                    'dataset': dataset,
                    'mention1_id': row['mention_id1'],
                    'mention2_id': row['mention_id2'],
                    'is_positive': is_positive,
                    'similarity': similarity
                })

                # Update summary counts
                key = (dataset, is_positive)
                summary_counts[key] = summary_counts.get(key, 0) + 1

        # Create filtered partition if it has data
        if filtered_rows:
            partition_num += 1
            filtered_df = pd.DataFrame(filtered_rows)
            partition_name = f"partition_{partition_num:05d}"
            filtered_partitions[partition_name] = filtered_df
            logger.info(f"  Partition {partition_key}: kept {len(filtered_df):,} pairs")

        del df, filtered_rows
        gc.collect()

    # Step 3: Print summary
    logger.info("\n" + "=" * 80)
    logger.info("FILTERED PAIRS SUMMARY")
    logger.info("=" * 80)

    total_filtered = sum(summary_counts.values())
    logger.info(f"Total filtered pairs: {total_filtered:,}")

    # Summary by dataset
    logger.info("\nBy dataset:")
    datasets = sorted(set(d for d, _ in summary_counts.keys()))
    for dataset in datasets:
        pos_count = summary_counts.get((dataset, True), 0)
        neg_count = summary_counts.get((dataset, False), 0)
        total = pos_count + neg_count
        logger.info(f"  {dataset}:")
        logger.info(f"    Positive (< 65th %ile): {pos_count:,}")
        logger.info(f"    Negative (> 96th %ile): {neg_count:,}")
        logger.info(f"    Total: {total:,}")

    logger.info("\nBy is_positive:")
    total_positive = sum(count for (_, is_pos), count in summary_counts.items() if is_pos)
    total_negative = sum(count for (_, is_pos), count in summary_counts.items() if not is_pos)
    logger.info(f"  Positive pairs (< 65th %ile): {total_positive:,}")
    logger.info(f"  Negative pairs (> 96th %ile): {total_negative:,}")

    logger.info("=" * 80)

    log_memory_usage("END filter_pairs_by_similarity_quantiles")

    return filtered_partitions


def sample_balanced_pairs_per_dataset(
    filtered_pairs_dict: Dict[str, pd.DataFrame],
    n_positive_per_dataset: int = 4545,
    n_negative_per_dataset: int = 45455,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Sample a balanced set of pairs per dataset.

    For each dataset:
    - Sample n_positive_per_dataset positive pairs (default: 4,545)
    - Sample n_negative_per_dataset negative pairs (default: 45,455)
    - Total: 50,000 pairs per dataset

    Args:
        filtered_pairs_dict: Dict of DataFrames (from PartitionedDataset) with columns:
                            dataset, mention1_id, mention2_id, is_positive, similarity
        n_positive_per_dataset: Number of positive pairs to sample per dataset
        n_negative_per_dataset: Number of negative pairs to sample per dataset
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled pairs: dataset, mention1_id, mention2_id, is_positive, similarity
    """
    logger.info(f"Sampling balanced pairs per dataset: {n_positive_per_dataset} positive, {n_negative_per_dataset} negative")
    log_memory_usage("START sample_balanced_pairs_per_dataset")

    np.random.seed(random_seed)

    # Step 1: Load all filtered pairs and group by dataset and is_positive
    logger.info("Step 1: Loading all filtered pairs")
    partition_keys = list(filtered_pairs_dict.keys())

    # Accumulate pairs by (dataset, is_positive)
    grouped_pairs = {}  # {(dataset, is_positive): [list of row dicts]}

    for partition_key in tqdm(partition_keys, desc="Loading filtered partitions"):
        partition_data = filtered_pairs_dict[partition_key]
        if callable(partition_data):
            df = partition_data()
        else:
            df = partition_data

        for (dataset, is_positive), group in df.groupby(['dataset', 'is_positive']):
            key = (dataset, is_positive)
            if key not in grouped_pairs:
                grouped_pairs[key] = []
            # Convert to list of dicts
            grouped_pairs[key].extend(group.to_dict('records'))

        del df
        gc.collect()

    log_memory_usage("AFTER loading all filtered pairs")

    # Step 2: Sample from each dataset
    logger.info("\nStep 2: Sampling pairs per dataset")
    sampled_rows = []
    datasets = sorted(set(d for d, _ in grouped_pairs.keys()))

    logger.info(f"\nTarget: {n_positive_per_dataset} positive + {n_negative_per_dataset} negative = {n_positive_per_dataset + n_negative_per_dataset} total per dataset")
    logger.info("-" * 80)

    for dataset in datasets:
        positive_key = (dataset, True)
        negative_key = (dataset, False)

        positive_pairs = grouped_pairs.get(positive_key, [])
        negative_pairs = grouped_pairs.get(negative_key, [])

        logger.info(f"\n{dataset}:")
        logger.info(f"  Available positive pairs: {len(positive_pairs):,}")
        logger.info(f"  Available negative pairs: {len(negative_pairs):,}")

        # Sample positive pairs
        if len(positive_pairs) >= n_positive_per_dataset:
            sampled_positive = np.random.choice(len(positive_pairs), size=n_positive_per_dataset, replace=False)
            sampled_positive_pairs = [positive_pairs[i] for i in sampled_positive]
            logger.info(f"  Sampled positive pairs: {len(sampled_positive_pairs):,}")
        else:
            logger.warning(f"  Not enough positive pairs! Only {len(positive_pairs):,} available, need {n_positive_per_dataset:,}")
            sampled_positive_pairs = positive_pairs
            logger.info(f"  Using all available positive pairs: {len(sampled_positive_pairs):,}")

        # Sample negative pairs
        if len(negative_pairs) >= n_negative_per_dataset:
            sampled_negative = np.random.choice(len(negative_pairs), size=n_negative_per_dataset, replace=False)
            sampled_negative_pairs = [negative_pairs[i] for i in sampled_negative]
            logger.info(f"  Sampled negative pairs: {len(sampled_negative_pairs):,}")
        else:
            logger.warning(f"  Not enough negative pairs! Only {len(negative_pairs):,} available, need {n_negative_per_dataset:,}")
            sampled_negative_pairs = negative_pairs
            logger.info(f"  Using all available negative pairs: {len(sampled_negative_pairs):,}")

        # Combine
        dataset_sampled = sampled_positive_pairs + sampled_negative_pairs
        sampled_rows.extend(dataset_sampled)
        logger.info(f"  Total sampled for {dataset}: {len(dataset_sampled):,}")

    # Create final DataFrame
    result_df = pd.DataFrame(sampled_rows)

    # Sort by dataset and is_positive for consistency
    result_df = result_df.sort_values(['dataset', 'is_positive'], ascending=[True, False]).reset_index(drop=True)

    # Step 3: Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLED PAIRS SUMMARY")
    logger.info("=" * 80)

    total_sampled = len(result_df)
    logger.info(f"Total sampled pairs: {total_sampled:,}")

    # By dataset
    logger.info("\nBy dataset:")
    for dataset in result_df['dataset'].unique():
        dataset_df = result_df[result_df['dataset'] == dataset]
        pos_count = len(dataset_df[dataset_df['is_positive']])
        neg_count = len(dataset_df[~dataset_df['is_positive']])
        total = len(dataset_df)
        logger.info(f"  {dataset}:")
        logger.info(f"    Positive: {pos_count:,}")
        logger.info(f"    Negative: {neg_count:,}")
        logger.info(f"    Total: {total:,}")

    # Overall
    total_positive = len(result_df[result_df['is_positive']])
    total_negative = len(result_df[~result_df['is_positive']])
    logger.info(f"\nOverall:")
    logger.info(f"  Total positive: {total_positive:,}")
    logger.info(f"  Total negative: {total_negative:,}")

    logger.info("=" * 80)

    log_memory_usage("END sample_balanced_pairs_per_dataset")

    return result_df


def compute_pairs_summary_stats(
    mention_pairs_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute summary statistics at dataset level.

    For each combination of (dataset, is_positive):
    - Count of pairs
    - Similarity quantiles: 0.25, 0.50 (median), 0.75
    - Mean similarity

    Args:
        mention_pairs_dict: Dict of DataFrames (from PartitionedDataset) with columns:
                           mention_id1, mention_id2, sentence_similarity, is_positive,
                           dataset

    Returns:
        DataFrame with columns: dataset, is_positive, count,
                               q25_similarity, q50_similarity, q75_similarity, mean_similarity
    """
    logger.info("Computing summary statistics at dataset level")
    log_memory_usage("START compute_pairs_summary_stats")

    # Process partitions one at a time and accumulate statistics
    partition_keys = list(mention_pairs_dict.keys())
    logger.info(f"Processing {len(partition_keys)} partitions for summary statistics")

    # Accumulate data for each group
    group_data = {}  # {(dataset, is_positive): [similarities]}

    for partition_key in tqdm(partition_keys, desc="Processing partitions for stats"):
        # Load partition
        partition_data = mention_pairs_dict[partition_key]
        if callable(partition_data):
            df = partition_data()
        else:
            df = partition_data

        # Group by (dataset, is_positive) and accumulate similarities
        for (dataset, is_positive), group in df.groupby(['dataset', 'is_positive']):
            key = (dataset, is_positive)
            if key not in group_data:
                group_data[key] = []
            group_data[key].extend(group['sentence_similarity'].tolist())

        # Free memory
        del df
        gc.collect()

    logger.info(f"Accumulated data for {len(group_data)} unique (dataset, is_positive) combinations")
    log_memory_usage("AFTER accumulating group data")

    # Compute statistics for each group
    summary_rows = []
    for (dataset, is_positive), similarities in tqdm(group_data.items(), desc="Computing statistics"):
        similarities_array = np.array(similarities)

        summary_rows.append({
            'dataset': dataset,
            'is_positive': is_positive,
            'count': len(similarities_array),
            'q25_similarity': np.percentile(similarities_array, 25),
            'q50_similarity': np.percentile(similarities_array, 50),  # median
            'q75_similarity': np.percentile(similarities_array, 75),
            'mean_similarity': np.mean(similarities_array)
        })

        # Free memory
        del similarities_array

    summary_df = pd.DataFrame(summary_rows)

    # Sort by dataset, is_positive
    summary_df = summary_df.sort_values(['dataset', 'is_positive']).reset_index(drop=True)

    # Log summary
    logger.info("\n" + "="*80)
    logger.info("PAIRS SUMMARY STATISTICS")
    logger.info("="*80)

    total_pairs = summary_df['count'].sum()
    logger.info(f"Total pairs across all groups: {total_pairs:,}")

    # Log by is_positive
    for is_pos in [True, False]:
        label = "Coreferential" if is_pos else "Non-coreferential"
        subset = summary_df[summary_df['is_positive'] == is_pos]
        count = subset['count'].sum()
        pct = (count / total_pairs * 100) if total_pairs > 0 else 0
        logger.info(f"\n{label} pairs: {count:,} ({pct:.1f}%)")

    # Log by dataset
    logger.info("\nBy dataset:")
    for dataset in summary_df['dataset'].unique():
        subset = summary_df[summary_df['dataset'] == dataset]
        count = subset['count'].sum()
        pct = (count / total_pairs * 100) if total_pairs > 0 else 0
        logger.info(f"  {dataset}: {count:,} ({pct:.1f}%)")

    logger.info("\nDetailed breakdown:")
    logger.info(summary_df.to_string())
    logger.info("="*80)

    log_memory_usage("END compute_pairs_summary_stats")

    return summary_df


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
