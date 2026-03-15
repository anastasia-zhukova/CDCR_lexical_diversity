# Pipeline 02: Pairs Selection - CORRECTED Understanding

## Critical Insight: Sentence-Level, Not Mention-Level

**Key Discovery**: The original code works at the **sentence level**, not mention level!

---

## Correct Flow from sampling.py

### Step 1: Encode Sentences (Not Mentions!)

```python
def encode_sentences(dict_sentences, model):
    # Input: All SENTENCES in a topic (not mentions)
    # Output: Sentence embeddings and pairwise similarities
```

**What gets encoded**: Full sentences that contain mentions
**NOT encoded**: Individual mentions or mention representations

### Step 2: Create Sentence Pairs

```python
def sentences_processing(dict_sentences_by_topic, mentions_by_sentence, model):
```

**Process**:
1. For each topic, encode **all sentences** with SBERT
2. Compute pairwise **sentence similarities** (lower triangle)
3. For each sentence pair `(sent1, sent2)`:
   - Check: Both sentences have mentions?
   - Check: `mentions_in_pair = mentions(sent1) ∪ mentions(sent2)` has at least 2 mentions?
   - If yes: Store `(sent1, sent2): sentence_similarity`
   - Also store: `product(mentions_sent1, mentions_sent2)` (all combinations)
4. Filter: Ignore sentence pairs with similarity = 1.0 (exact duplicates)

**Key Point**: Similarity is computed at **sentence level**, then applied to all mention pairs from those sentences!

### Step 3: Create Mention Pairs from Sentence Pairs

```python
def create_mention_pairs(dict_sentences_pairs_similarity, mentions,
                        mention_pairs_by_pair, ...):
    for topic, sentence_pairs in dict_sentences_pairs_similarity.items():
        topic_sentence_similarity_median = median(sentence_pairs.values())

        for (sent1, sent2), sentence_similarity in sentence_pairs.items():
            # Get all mention combinations from this sentence pair
            for mention1, mention2 in product(mentions_sent1, mentions_sent2):
                # Assign SENTENCE similarity to this mention pair
                same_coref = (mention1.coref_chain == mention2.coref_chain)

                if same_coref:
                    positive_pairs[(mention1, mention2)] = sentence_similarity
                else:
                    negative_pairs[(mention1, mention2)] = sentence_similarity
```

**Critical**: Each mention pair `(m1, m2)` gets the **sentence similarity** from `(sent1, sent2)`, not its own similarity!

### Step 4: Hard Mining with Sentence Similarities

```python
# Filter based on SENTENCE-level similarities
topic_sentence_median = median(all_sentence_similarities_in_topic)
q3_positive = percentile(positive_pair_similarities, 75)

# Hard positives: Same coref, LOW sentence similarity
hard_positives = {(m1,m2): sim for (m1,m2), sim in positives.items()
                  if sim < topic_sentence_median}

# Hard negatives: Different coref, HIGH sentence similarity
hard_negatives = {(m1,m2): sim for (m1,m2), sim in negatives.items()
                  if sim > q3_positive}
```

**Insight**:
- Hard positives: Coreferent mentions from **dissimilar sentences**
- Hard negatives: Non-coreferent mentions from **similar sentences**

---

## Why This Approach?

### Advantages of Sentence-Level Similarity

1. **Computational Efficiency**
   - Encode N sentences, not N×M mentions
   - If topic has 100 sentences with 300 mentions:
     - Sentence-level: 100 encodings
     - Mention-level: 300 encodings

2. **Broader Context**
   - Sentence similarity captures overall semantic similarity
   - Mentions from similar sentences are contextually related
   - Even if specific mentions differ

3. **Natural Hard Mining**
   - Hard positives: Same entity mentioned in different contexts (dissimilar sentences)
   - Hard negatives: Different entities in similar contexts (confusing cases)

### Example

**Sentence 1**: "President Biden announced new climate policies."
**Sentence 2**: "The President unveiled his environmental agenda."

- Sentence similarity: HIGH (similar topics, similar structure)
- Mention pairs from these sentences:
  - ("Biden", "President") → Positive, HIGH similarity → FILTERED OUT (too easy)
  - ("Biden", "agenda") → Negative, HIGH similarity → KEPT (hard negative - similar context but different entities)

---

## Corrected Pipeline 02 Design

### Input Requirements

From pipeline 01, we need:
1. **mentions_for_pairs.csv** (train split only)
   - mention_id, coref_chain, topic, entity_style, split
   - conll_doc_key, sent_id (to identify which sentence contains each mention)

2. **Raw sentence data** (NOT mention representations!)
   - We need the original sentences (without [E]/[/E] markers)
   - Group mentions by sentence
   - Group sentences by topic

### Node 1: Prepare Sentences by Topic

**Function**: `prepare_sentences_by_topic()`

```python
def prepare_sentences_by_topic(
    mentions_df: pd.DataFrame,  # mentions_for_pairs.csv
    data_path: str              # path to raw documents
) -> Tuple[Dict, Dict]:
    """
    Prepare sentences grouped by topic and map mentions to sentences.

    Returns:
        sentences_by_topic: {topic: {sentence_id: sentence_text}}
        mentions_by_sentence: {sentence_id: {mention_ids}}
    """
    # 1. Filter to train split
    # 2. Load raw documents
    # 3. Extract sentences (full text, no markers)
    # 4. Group by topic
    # 5. Map mentions to their sentences
```

**Output**:
- `sentences_by_topic` (Dict or JSON)
- `mentions_by_sentence` (Dict or JSON)

### Node 2: Encode Sentences with SBERT

**Function**: `encode_sentences_by_topic()`

```python
def encode_sentences_by_topic(
    sentences_by_topic: Dict,
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> pd.DataFrame:
    """
    Encode sentences (not mentions) and compute pairwise similarities.

    For each topic:
    1. Encode all sentences
    2. Compute pairwise cosine similarity
    3. Store sentence pairs with similarities

    Returns:
        DataFrame: sentence_id1, sentence_id2, similarity, topic
    """
```

**Output**: `sentence_pairs_with_similarity.parquet`

### Node 3: Create Mention Pairs from Sentence Pairs

**Function**: `create_mention_pairs_from_sentences()`

```python
def create_mention_pairs_from_sentences(
    sentence_pairs_df: pd.DataFrame,
    mentions_by_sentence: Dict,
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

    Returns:
        DataFrame: mention_id1, mention_id2, sentence_similarity,
                   is_positive, topic
    """
```

**Output**: `mention_pairs_from_sentences.parquet`

### Node 4: Select Hard Pairs

**Function**: `select_hard_pairs_sentence_based()`

```python
def select_hard_pairs_sentence_based(
    mention_pairs_df: pd.DataFrame,
    additional_sampling: bool = True
) -> pd.DataFrame:
    """
    Select hard pairs based on SENTENCE similarities.

    For each topic:
    1. Compute topic sentence similarity median
    2. Hard positives: is_positive=1 AND sentence_sim < topic_median
    3. Hard negatives: is_positive=0 AND sentence_sim > Q3(positive_sims)
    4. Optional: Sample 6*sqrt(n_mentions) positives, 15x negatives

    Returns:
        DataFrame: Selected pairs with labels
    """
```

**Output**: `selected_pairs.parquet`

---

## Key Differences from Previous Design

| Aspect | Previous (Wrong) | Corrected |
|--------|-----------------|-----------|
| What to encode | Mention representations | Full sentences |
| Similarity computed on | Individual mentions | Sentence pairs |
| Mention pairs created by | Direct pairing | Product of mentions in sentence pairs |
| Similarity value | Mention-to-mention | Sentence-to-sentence (shared) |
| Hard mining based on | Mention similarity | Sentence similarity |

---

## Data Requirements

### What We Have (from pipeline 01)
✅ `mentions_for_pairs.csv` with:
- mention_id
- coref_chain
- topic, entity_style
- conll_doc_key, sent_id (KEY: identifies which sentence)
- tokens_number

✅ Raw document parquet files

### What We Need to Extract
1. **Sentence text** (full, without [E]/[/E] markers)
   - Identify by: (conll_doc_key, sent_id)
   - Group by: topic

2. **Mentions per sentence**
   - Map: sentence_id → {mention_ids}

---

## Expected Pair Generation

### Example

**Topic**: "COVID-19 Vaccines"

**Sentence 1** (ID: s1): "Pfizer announced vaccine trial results."
- Mentions: m1="Pfizer", m2="vaccine", m3="trial results"

**Sentence 2** (ID: s2): "Moderna reported similar efficacy data."
- Mentions: m4="Moderna", m5="efficacy data"

**Sentence 3** (ID: s3): "Pfizer's CEO discussed production plans."
- Mentions: m6="Pfizer", m7="CEO", m8="production plans"

**Process**:
1. Encode sentences: s1, s2, s3
2. Compute similarities:
   - sim(s1, s2) = 0.85 (high - similar topics)
   - sim(s1, s3) = 0.65 (medium)
   - sim(s2, s3) = 0.45 (low)

3. Create mention pairs from sentence pairs:
   - From (s1, s2, sim=0.85):
     - (m1, m4) → Negative, sim=0.85 → Hard negative (similar context, diff entities)
     - (m2, m5) → Positive, sim=0.85 → Filtered out (too easy)
     - ... (all combinations)

   - From (s1, s3, sim=0.65):
     - (m1, m6) → Positive, sim=0.65 → Could be kept if < topic_median
     - (m2, m8) → Negative, sim=0.65 → Filtered out (not high enough)
     - ...

---

## Implementation Checklist

- [ ] Extract sentences from raw documents (no markers)
- [ ] Group sentences by topic
- [ ] Map mentions to sentences
- [ ] Encode sentences with SBERT
- [ ] Compute sentence-pair similarities
- [ ] Create mention pairs via product()
- [ ] Assign sentence similarities to mention pairs
- [ ] Apply hard mining filters
- [ ] Optional sampling (6*sqrt(n))

---

## Critical Insight Summary

The original code is **sentence-centric**, not **mention-centric**:

1. **Unit of similarity**: Sentences
2. **Pairing strategy**: Find similar sentence pairs, then create all mention combinations
3. **Hard mining**: Based on sentence similarity, not mention similarity
4. **Efficiency**: Encode fewer items (sentences < mentions)
5. **Rationale**: Mentions in similar sentences share context, making negative pairs harder

This is actually a **smarter approach** than mention-level because:
- More efficient computationally
- Captures contextual similarity
- Natural hard negative mining (similar contexts, different entities)

---

**Next Step**: Implement pipeline 02 with this CORRECTED understanding!
