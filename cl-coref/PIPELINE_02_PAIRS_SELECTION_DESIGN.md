# Pipeline 02: Pairs Selection - Design Document

## Overview

This pipeline implements the hard negative mining strategy from the reference paper to select training pairs.

**Input**: Train mentions with string representations (from pipeline 01)
**Output**: Selected pairs (positive and negative) for contrastive learning

---

## Original Strategy Analysis (from sampling.py)

### Step 1: Encode Sentences with SBERT
```python
def encode_sentences(dict_sentences, model):
    # Encode all sentences in a topic using SBERT
    # Compute pairwise cosine similarity
    # Return similarity matrix (lower triangle only)
```

**Purpose**: Get sentence-level similarity scores to find hard positives/negatives

### Step 2: Process Sentences by Topic
```python
def sentences_processing(dict_sentences_by_topic, mentions_by_sentence, model):
```

**Process**:
1. For each topic, encode all sentences with SBERT
2. Compute pairwise cosine similarities (lower triangle)
3. For each sentence pair:
   - Check if both sentences have mentions
   - Get mentions from both sentences
   - Create all mention pairs using `product(mentions_s1, mentions_s2)`
   - Filter: Ignore pairs with similarity = 1.0 (exact duplicates)
   - Store: `{(sentence1, sentence2): similarity_score}`

**Output**:
- `dict_sentences_pair_similarity_by_topic`: Sentence pairs with scores
- `dict_mentions_by_topic`: All mentions involved in pairs
- `dict_mention_pairs_by_pair`: Mention combinations per sentence pair

### Step 3: Create Mention Pairs
```python
def create_mention_pairs(dict_sentences_pairs_similarity, mentions,
                        mention_pairs_by_pair, mention_keys_by_topic_in_type,
                        additional_sampling=False):
```

**Process**:

#### 3.1: Split Positive/Negative Pairs
- For each mention pair:
  - **Positive**: Same coref_chain
  - **Negative**: Different coref_chain
- Store with similarity scores

#### 3.2: Hard Negative Mining (First Stage)
```python
# Positive pairs: Keep those with similarity < topic_median
topic_sentence_similarity_median = statistics.median(sentence_pairs.values())
positive_pairs = {k:v for k,v in positives.items() if v < topic_median}

# Negative pairs: Keep those with similarity > Q3 of positive pairs
third_quartile = np.percentile(positive_pairs_scores, 75)
negative_pairs = {k:v for k,v in negatives.items() if v > third_quartile}
```

**Rationale**:
- **Hard Positives**: Coreferent mentions with LOW similarity (harder to learn)
- **Hard Negatives**: Non-coreferent mentions with HIGH similarity (harder to distinguish)

#### 3.3: Additional Sampling (Optional)
```python
if additional_sampling:
    positive_limit = int(6 * sqrt(num_mentions_in_topic))
    negative_limit = positive_limit * 15

    # Random sample if exceeds limits
    if len(positives) > positive_limit:
        positives = random.sample(positives, positive_limit)
    if len(negatives) > negative_limit:
        negatives = random.sample(negatives, negative_limit)
```

**Rationale**: Limit pairs per topic to balance training

---

## Key Differences from Reference Paper

### Reference Paper (Caciularu et al., 2021):
- Uses first 2 sentences as document context
- SBERT encodes: `context + [SEP] + mention_sentence`
- Positive pairs: Least similar coreferent mentions
- Negative pairs: Most similar non-coreferent mentions

### Original Implementation (sampling.py):
- Uses SBERT to encode **full sentences** (not context + mention)
- Computes **sentence-level similarity** first
- Then creates mention pairs from similar sentences
- Same hard mining strategy (low sim positives, high sim negatives)

### Our Implementation (will use):
- Use **string representations** from pipeline 01 (8 sentences context + mention)
- Encode with SBERT: `"{8 sentences} [SEP] {[E] mention [/E]}"`
- Compute **mention-level similarity** directly (not sentence-level)
- Apply hard mining to mention pairs

**Advantage**: More aligned with training objective (mention representations, not sentences)

---

## Pipeline 02 Structure

### Inputs (from pipeline 01)

1. **mention_representations.parquet**
   - Columns: mention_id, string_representation, document_context, mention_context
   - Filter to: train split only

2. **mentions_for_pairs.csv**
   - Need: mention_id, coref_chain, topic, entity_style, split
   - Filter to: train split only

### Node 1: Encode Mentions with SBERT

**Function**: `encode_train_mentions()`

```python
def encode_train_mentions(
    mention_representations_df: pd.DataFrame,
    mentions_for_pairs_df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> pd.DataFrame:
    """
    Encode mention string representations using SBERT.

    Returns:
        DataFrame with: mention_id, embedding (768-dim vector), coref_chain, topic
    """
```

**Process**:
1. Filter both dataframes to train split only
2. Merge on mention_id to get string_representation + metadata
3. Load SBERT model
4. Encode all string representations
5. Return: mention_id, embedding, coref_chain, topic, entity_style

**Output**: `train_mention_embeddings` (MemoryDataset or Parquet)

### Node 2: Compute Pairwise Similarities

**Function**: `compute_mention_similarities()`

```python
def compute_mention_similarities(
    mention_embeddings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity for mentions within each topic.

    Returns:
        DataFrame with: mention_id1, mention_id2, similarity, topic,
                       coref_chain1, coref_chain2, is_positive
    """
```

**Process**:
1. Group embeddings by topic (and optionally by entity_style)
2. For each topic:
   - Compute pairwise cosine similarity (lower triangle)
   - Create pairs: (mention_id1, mention_id2, similarity)
   - Label: is_positive = (coref_chain1 == coref_chain2)
3. Filter: Remove pairs with similarity = 1.0 (duplicates)
4. Concatenate all topics

**Output**: `mention_pairs_with_similarity` (Parquet)

### Node 3: Select Hard Pairs

**Function**: `select_hard_pairs()`

```python
def select_hard_pairs(
    pairs_df: pd.DataFrame,
    additional_sampling: bool = True
) -> pd.DataFrame:
    """
    Select hard positive and negative pairs using similarity thresholds.

    Strategy:
    - Hard Positives: is_positive=1 AND similarity < topic_median
    - Hard Negatives: is_positive=0 AND similarity > Q3(positive_similarities)

    Optional: Limit pairs per topic (6*sqrt(n_mentions) positives, 15x negatives)

    Returns:
        DataFrame with selected pairs and labels
    """
```

**Process**:
1. Group by topic
2. For each topic:
   - Split into positive_pairs and negative_pairs
   - Compute topic sentence similarity median
   - Compute Q3 of positive pair similarities
   - **Hard Positive Filter**: similarity < topic_median
   - **Hard Negative Filter**: similarity > Q3
3. Optional additional sampling:
   - positive_limit = 6 * sqrt(num_mentions_in_topic)
   - negative_limit = 15 * positive_limit
   - Random sample if exceeds limits
4. Combine all topics
5. Shuffle pairs

**Output**: `selected_pairs` (Parquet)
- Columns: mention_id1, mention_id2, label (1=positive, 0=negative), similarity, topic

### Node 4: Create Pairs Statistics

**Function**: `compute_pairs_statistics()`

```python
def compute_pairs_statistics(
    selected_pairs_df: pd.DataFrame
) -> dict:
    """
    Compute statistics about selected pairs.

    Returns:
        Statistics JSON with:
        - Total pairs, positive pairs, negative pairs
        - Ratio positive/negative
        - Per-topic breakdown
        - Similarity distributions
    """
```

**Output**: `pairs_statistics.json`

---

## Data Catalog Configuration

```yaml
# conf/base/catalog_02_pairs_selection.yml

# Train mention embeddings (intermediate)
train_mention_embeddings:
  type: pandas.ParquetDataset
  filepath: data/03_primary/train_mention_embeddings.parquet
  save_args:
    engine: pyarrow
    compression: snappy

# All mention pairs with similarities
mention_pairs_with_similarity:
  type: pandas.ParquetDataset
  filepath: data/03_primary/mention_pairs_with_similarity.parquet
  save_args:
    engine: pyarrow
    compression: snappy

# Selected hard pairs (final output)
selected_pairs:
  type: pandas.ParquetDataset
  filepath: data/04_model_input/selected_pairs.parquet
  save_args:
    engine: pyarrow
    compression: snappy

# Pairs statistics
pairs_statistics:
  type: json.JSONDataset
  filepath: data/08_reporting/pairs_statistics.json
```

---

## Parameters Configuration

```yaml
# conf/base/parameters_02_pairs_selection.yml

# SBERT model for encoding
sbert_model_name: "sentence-transformers/all-mpnet-base-v2"

# Hard mining strategy
hard_mining:
  # Positive pairs: Keep those with similarity < topic_median
  positive_threshold: "median"  # or specific value like 0.5

  # Negative pairs: Keep those with similarity > Q3 of positive pairs
  negative_threshold: "q3"  # or specific value like 0.75

  # Filter exact duplicates
  filter_similarity_one: true

# Additional sampling limits
additional_sampling:
  enabled: true
  positive_limit_formula: "6 * sqrt(n_mentions)"  # Per topic
  negative_multiplier: 15  # negative_limit = positive_limit * 15

# Topic grouping
group_by:
  - topic
  - entity_style  # Optional: group by entity_style within topics

# Random seed for reproducibility
random_seed: 42
```

---

## Expected Output Statistics

Based on original implementation:

```
Train mentions: ~74,611
Topics: ~45
Entity styles: 2 (entity, event)

Estimated pairs:
- Before filtering: ~2.5M pairs
- After hard mining: ~500K-1M pairs
  - Positives: ~50K-100K (10-20%)
  - Negatives: ~450K-900K (80-90%)
- After additional sampling: ~100K-200K pairs
  - Positives: ~10K-20K
  - Negatives: ~90K-180K

Per-topic distribution:
- Median positive pairs per topic: ~500-1000
- Median negative pairs per topic: ~7500-15000
```

---

## Implementation Order

1. **Create pipeline structure**: `src/cl_coref/pipelines/02_pairs_selection/`
2. **Implement Node 1**: Encode mentions with SBERT
3. **Implement Node 2**: Compute pairwise similarities
4. **Implement Node 3**: Select hard pairs
5. **Implement Node 4**: Compute statistics
6. **Test pipeline**: Run on train split
7. **Validate output**: Check pair distributions

---

## Key Decisions

### 1. Encoding Strategy
**Decision**: Use full string representations (8 sentences + mention with [E]/[/E])

**Alternative**: Encode only mention sentences (like original sampling.py)

**Rationale**: Training will use full context, so pair selection should match

### 2. Grouping Strategy
**Decision**: Group by (topic, entity_style)

**Alternative**: Group by topic only

**Rationale**: Entity and event mentions may have different similarity distributions

### 3. Additional Sampling
**Decision**: Enable by default with 6*sqrt(n) formula

**Alternative**: Keep all hard pairs without sampling

**Rationale**: Balance training across topics, prevent large topics from dominating

### 4. Similarity Threshold
**Decision**: Use topic-specific median and Q3

**Alternative**: Use global thresholds

**Rationale**: Different topics may have different similarity distributions

---

## Next Steps

1. Implement pipeline 02_pairs_selection
2. Test on small subset first
3. Run on full train split
4. Validate pair quality and distributions
5. Use for model training in pipeline 03

---

**Status**: Design Complete - Ready for Implementation
**Estimated Development Time**: 3-4 hours
**Estimated Execution Time**: ~30-60 minutes (SBERT encoding is slow)
