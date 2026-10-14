# Pipeline 02: Pairs Selection - Implementation

**Date**: February 9, 2026
**Status**: Ready to Run

---

## Overview

Pipeline 02 implements sentence-level similarity-based pair selection following the original sampling.py strategy. This approach is more efficient and creates naturally hard negative pairs by comparing mentions from similar sentence contexts.

---

## Pipeline Architecture

### 4 Nodes

1. **prepare_sentences_by_topic**: Extract raw sentences grouped by topic
2. **encode_sentences_by_topic**: Encode sentences with SBERT and compute similarities
3. **create_mention_pairs_from_sentences**: Create mention pairs via product()
4. **select_hard_pairs_sentence_based**: Apply hard mining filters

---

## Node Details

### Node 1: prepare_sentences_by_topic

**Purpose**: Extract and organize sentences for encoding

**Process**:
1. Filter mentions to train split only (74,611 mentions)
2. Load raw documents from parquet files
3. Extract full sentences (WITHOUT [E]/[/E] markers)
4. Group sentences by topic
5. Map mentions to their sentences

**Inputs**:
- `mentions_for_pairs` (CSV from pipeline 01)
- `params:data_path` (path to raw parquet files)

**Outputs**:
- `sentences_by_topic`: Dict[topic → Dict[sentence_id → sentence_text]]
- `mentions_by_sentence`: Dict[sentence_id → Set[mention_ids]]

**Implementation Notes**:
- Uses batch processing for large parquet files (1M rows per batch)
- Creates unique sentence IDs: `{dataset}_{doc_key}_{sent_id}`
- Caches loaded documents to avoid repeated reads

---

### Node 2: encode_sentences_by_topic

**Purpose**: Encode sentences and compute pairwise similarities

**Process**:
1. Load SBERT model (`all-mpnet-base-v2`)
2. For each topic:
   - Encode all sentences
   - Compute pairwise cosine similarity (lower triangle only)
   - Filter to sentence pairs where both have mentions
   - Filter to pairs with total mentions >= 2
   - Filter out exact duplicates (similarity ≈ 1.0)

**Inputs**:
- `sentences_by_topic` (from node 1)
- `mentions_by_sentence` (from node 1)
- `params:sbert_model_name` (default: "sentence-transformers/all-mpnet-base-v2")

**Outputs**:
- `sentence_pairs_with_similarity`: DataFrame
  - Columns: sentence_id1, sentence_id2, similarity, topic

**Implementation Notes**:
- Uses `convert_to_tensor=True` for GPU acceleration if available
- Only computes lower triangle to avoid duplicate pairs
- Filters exact duplicates (similarity >= 0.9999)

---

### Node 3: create_mention_pairs_from_sentences

**Purpose**: Create all mention combinations from sentence pairs

**Process**:
1. For each sentence pair (sent1, sent2) with similarity:
   - Get all mentions from sent1
   - Get all mentions from sent2
   - Create Cartesian product: product(mentions1, mentions2)
   - Assign SENTENCE similarity to all mention pairs
   - Label: `is_positive = (coref_chain1 == coref_chain2)`

**Inputs**:
- `sentence_pairs_with_similarity` (from node 2)
- `mentions_by_sentence` (from node 1)
- `mentions_for_pairs` (to lookup coref_chain)

**Outputs**:
- `mention_pairs_from_sentences`: DataFrame
  - Columns: mention_id1, mention_id2, sentence_similarity, is_positive, topic

**Key Insight**: Each mention pair gets the SENTENCE similarity, not its own similarity!

---

### Node 4: select_hard_pairs_sentence_based

**Purpose**: Apply hard mining filters

**Process**:
For each topic:
1. Compute topic sentence similarity median
2. Compute Q3 (75th percentile) of positive pair similarities
3. **Hard positives**: is_positive=True AND sentence_sim < topic_median
   - Same coreference, LOW sentence similarity
   - Mentions from dissimilar contexts
4. **Hard negatives**: is_positive=False AND sentence_sim > Q3(positive)
   - Different coreference, HIGH sentence similarity
   - Mentions from similar contexts (confusing cases)

**Inputs**:
- `mention_pairs_from_sentences` (from node 3)
- `params:additional_sampling` (default: false)

**Outputs**:
- `selected_pairs`: DataFrame
  - Columns: mention_id1, mention_id2, sentence_similarity, is_positive, topic

**Implementation Notes**:
- `additional_sampling` is disabled initially to see full distribution
- Will implement sampling strategy later after analyzing pair counts

---

## Configuration

### catalog_02_pairs_selection.yml

```yaml
# Intermediate outputs
sentences_by_topic: MemoryDataset
mentions_by_sentence: MemoryDataset

sentence_pairs_with_similarity:
  type: pandas.ParquetDataset
  filepath: data/04_feature/sentence_pairs_with_similarity.parquet

mention_pairs_from_sentences:
  type: pandas.ParquetDataset
  filepath: data/04_feature/mention_pairs_from_sentences.parquet

# Final output
selected_pairs:
  type: pandas.ParquetDataset
  filepath: data/05_model_input/selected_pairs.parquet
```

### parameters_02_pairs_selection.yml

```yaml
# SBERT model for encoding sentences
sbert_model_name: "sentence-transformers/all-mpnet-base-v2"

# Sampling disabled initially to see full distribution
additional_sampling: false
```

---

## Running the Pipeline

### Prerequisites

Ensure pipeline 01 has completed successfully:
- `data/03_primary/mentions_for_pairs.csv` exists
- Contains 116,793 mentions (train + val + test)

### Command

```bash
cd /home/ubuntu/github/CDCR_lexical_diversity/cl-coref
kedro run --pipeline 02_pairs_selection
```

### Expected Runtime

Estimated: 2-4 hours (depends on GPU availability)

Breakdown:
- Node 1 (prepare_sentences): ~30-45 min (loading documents)
- Node 2 (encode_sentences): ~1-2 hours (SBERT encoding)
- Node 3 (create_mention_pairs): ~30-45 min (product combinations)
- Node 4 (select_hard_pairs): ~10-20 min (filtering)

---

## Expected Outputs

### sentence_pairs_with_similarity.parquet

Sample:
```
sentence_id1                      sentence_id2                      similarity  topic
ecb_plus_doc1_sent5              ecb_plus_doc2_sent12              0.723       COVID-19 vaccines
ecb_plus_doc1_sent5              ecb_plus_doc3_sent8               0.456       COVID-19 vaccines
```

### mention_pairs_from_sentences.parquet

Sample:
```
mention_id1  mention_id2  sentence_similarity  is_positive  topic
m123         m456         0.723                True         COVID-19 vaccines
m123         m789         0.723                False        COVID-19 vaccines
```

### selected_pairs.parquet (FINAL)

Sample:
```
mention_id1  mention_id2  sentence_similarity  is_positive  topic
m123         m456         0.421                True         COVID-19 vaccines    # Hard positive
m123         m789         0.856                False        COVID-19 vaccines    # Hard negative
```

---

## Key Statistics to Monitor

### After Node 1
- Total unique sentences extracted
- Sentences per topic
- Mentions per sentence (average)

### After Node 2
- Total sentence pairs
- Sentence pairs per topic
- Similarity distribution (median, Q1, Q3)
- Filtered duplicates count

### After Node 3
- Total mention pairs created
- Positive vs negative ratio
- Pairs per topic

### After Node 4
- Hard positives selected (% of total positives)
- Hard negatives selected (% of total negatives)
- Final positive/negative ratio
- Topic-level median similarities
- Q3 positive similarities per topic

---

## Validation Checks

1. **Sentence extraction**:
   - All sentences should have NO [E]/[/E] markers
   - Sentence IDs should be unique

2. **Similarity computation**:
   - All similarities should be in range [-1, 1]
   - No self-similarities (sent_id1 != sent_id2)
   - No exact duplicates (similarity < 0.9999)

3. **Mention pairs**:
   - Every pair should have is_positive label
   - mention_id1 and mention_id2 should exist in mentions_for_pairs
   - Pairs should come from different sentences

4. **Hard mining**:
   - Hard positives should have similarity < topic_median
   - Hard negatives should have similarity > Q3(positives)
   - Both positive and negative pairs should exist for each topic

---

## Troubleshooting

### Out of Memory Error (Node 1)

**Symptom**: Process killed when loading documents

**Solution**: Batch processing is already implemented with 1M rows per batch. If still failing, reduce batch size in nodes.py:

```python
for batch in parquet_file.iter_batches(batch_size=500_000):  # Reduce from 1M
```

### SBERT Model Download (Node 2)

**Symptom**: Model download times out or fails

**Solution**: Pre-download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
```

### Too Many Pairs (Node 3)

**Symptom**: Memory error when creating mention pairs

**Solution**: This is expected for topics with many mentions. The pipeline will save to disk incrementally. If it fails, we can enable sampling in node 4.

---

## Next Steps

After this pipeline completes:

1. **Analyze pair distribution**:
   - Check `selected_pairs.parquet` statistics
   - Plot similarity distributions by topic
   - Verify positive/negative ratios

2. **Decide on sampling**:
   - If too many pairs (>10M), enable sampling
   - Implement sampling strategy based on analysis

3. **Create pipeline 03: Tokenization**:
   - Tokenize selected pairs
   - Create model-ready inputs

---

## Comparison with Original Code

| Aspect | Original (sampling.py) | Pipeline 02 |
|--------|------------------------|-------------|
| Unit of similarity | Sentences | Sentences ✓ |
| Encoding model | SBERT | SBERT ✓ |
| Pair creation | product() | product() ✓ |
| Similarity assignment | Sentence-level | Sentence-level ✓ |
| Hard positives | sim < topic_median | sim < topic_median ✓ |
| Hard negatives | sim > Q3(positives) | sim > Q3(positives) ✓ |
| Sampling | 6*sqrt(n) | Disabled initially |

---

**Status**: ✅ Implementation complete
**Next Action**: Run the pipeline and analyze results
