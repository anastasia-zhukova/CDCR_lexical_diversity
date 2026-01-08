# CL_COREF: Contrastive Learning for Cross-Document Coreference Resolution

This module implements a siamese neural network trained with contrastive learning for cross-document event and entity coreference resolution.

## Table of Contents
- [System Overview](#system-overview)
- [Data Structures](#data-structures)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Expected Results](#expected-results)

---

## System Overview

The system learns to represent mentions (events or entities) in an embedding space where coreferent mentions (mentions referring to the same real-world event/entity) are close together and non-coreferent mentions are far apart.

**Key Components:**
- **CDCR_Dataset**: Loads and structures raw CoNLL data containing documents, sentences, and mentions
- **Sampling Module**: Generates positive/negative mention pairs based on sentence similarity
- **CL_Dataset**: PyTorch Dataset that tokenizes mentions and provides training pairs
- **Siamese Model**: Twin neural networks that encode mentions into embeddings
- **Clustering**: Groups mention embeddings to form coreference chains

---

## Data Structures

### Mention
A `Mention` object represents a single event or entity mention in a document. Each mention contains:

**Core Attributes:**
- `mention_id`: Unique identifier
- `type`: "event" or "entity"
- `coref_chain`: Gold standard coreference cluster ID
- `tokens_text`: List of tokens in the mention
- `tokens_str`: String representation of the mention
- `mention_head`: Head word of the mention
- `mention_head_lemma`: Lemmatized head word

**Context Information:**
- `sent_id`: Sentence ID where the mention appears
- `doc_id`: Document ID
- `mention_context`: Sentence containing the mention
- `new_context`: Tuple of (document_context, mention_sentence) used for model input

**Metadata:**
- `topic_id`, `subtopic_id`: Topic clustering information
- `is_singleton`: Whether the mention appears only once in its coreference chain
- `conll_doc_key`, `conll_sent_key`: Keys for CoNLL format alignment

### CDCR_Dataset
Organizes all mentions from one or more CoNLL datasets.

**Main Attributes:**
- `mentions`: Dictionary mapping mention_id → Mention object
- `mentions_by_type`: Dictionary with keys "event"/"entity" → set of mention IDs
- `mentions_by_sentence`: Maps sentence_key → set of mention IDs in that sentence
- `mentions_by_topic`: Maps topic_id → list of mention IDs
- `sentences_by_topic`: Maps topic_id → dictionary of sentence texts
- `gold_clusters_df`: DataFrame with gold coreference cluster assignments

**Purpose:** Provides structured access to all mentions and their relationships for sampling and evaluation.

### CL_Dataset (PyTorch Dataset)
Wraps a CDCR_Dataset and prepares data for model training/inference.

**Initialization Process:**
1. Creates a CDCR_Dataset from CoNLL data paths
2. **Tokenizes all mentions upfront** (not per-batch) using the specified transformer tokenizer
3. For **training mode**: Generates positive/negative mention pairs via the sampling module
4. For **inference mode**: Creates a list of all mention keys for evaluation

**Tokenization Details:**
- Uses transformer tokenizer (e.g., RoBERTa, BART)
- Input format: (document_context, mention_sentence_with_markers)
- Special tokens (e.g., `[E]`, `[/E]`) mark mention boundaries
- Max length: 120 tokens (truncates document context if needed)
- Padding: "max_length" for uniform batch processing
- **All mentions are tokenized once during initialization** for efficiency

---

## Data Processing Pipeline

### Training Data Flow

```
CoNLL Files
    ↓
CDCR_Dataset (loads mentions, documents, sentences)
    ↓
Sampling Module (return_pairs_and_labels)
    ↓
    1. Sentence Similarity Computation
       - Groups sentences by topic
       - Encodes sentences with SentenceTransformer (all-mpnet-base-v2)
       - Computes pairwise cosine similarity within topics
    ↓
    2. Mention Pair Generation
       - For each sentence pair, generates all mention pairs
       - Labels pairs: 1 (same coref_chain) or 0 (different coref_chain)
    ↓
    3. Two-Stage Subsampling
       Stage 1 - Similarity-based filtering:
         • Positive pairs: Keep pairs with sentence similarity < topic_median
         • Negative pairs: Keep pairs with sentence similarity > Q3(positive_similarities)

       Stage 2 - Count-based limiting (if --additional_sampling):
         • Positive limit: 6 * sqrt(num_mentions_in_topic)
         • Negative limit: 15 * positive_limit
    ↓
CL_Dataset (training pairs + labels)
    ↓
DataLoader (batches for training)
    ↓
Siamese Model Training
```

**Key Insight:** The sampling strategy focuses on **hard negative examples** (sentences that are similar but contain non-coreferent mentions) and **hard positive examples** (sentences that are dissimilar but contain coreferent mentions) to improve model discrimination.

### Inference Data Flow

```
CoNLL Files
    ↓
CDCR_Dataset
    ↓
CL_Dataset (all mentions, inference mode)
    ↓
DataLoader
    ↓
Siamese Model (generates embeddings)
    ↓
Clustering (Agglomerative with cosine distance)
    ↓
Predicted Coreference Chains
    ↓
Evaluation Metrics (CoNLL, LEA, PD_LEA, etc.)
```

---

## Training

### Entry Point: `train.py`

Train a siamese model for cross-document coreference resolution.

### Command-Line Arguments

#### Essential Parameters

**Data Configuration:**
- `--data_setup` (int, default=0) - **Critical**
  - `0`: ECBplus only (standard benchmark)
  - `1`: ECBplus + FCC-T (cross-dataset training)
  - `2`: ECBplus + NP4E (cross-dataset training)

- `--mention_type` (str, default='event') - **Critical**
  - `event`: Train on event mentions
  - `entity`: Train on entity mentions

**Model Selection:**
- `--language_model` (str, default='roberta-large') - **Critical**
  - Transformer backbone (e.g., `roberta-large`, `bart-large`, `deberta-large`)
  - Best results achieved with: `bart-large`

**Training Hyperparameters:**
- `--num_epochs` (int, default=5) - **Important**

- `--train_batch_size` (int, default=32) - **Important**
  - Recommended: 32 (balance between speed and memory)

- `--learning_rate` (float, default=2e-7) - **Important**
  - Recommended: 2e-7 for large models

- `--margin` (float, default=0.4) - **Important**
  - Contrastive loss margin
  - Recommended: 0.3 (from hyperparameter tuning)

- `--clustering_threshold` (float, default=0.2) - **Critical**
  - Cosine distance threshold for agglomerative clustering
  - Recommended: 0.2
  - Used during validation and saved in model filename

**Evaluation:**
- `--metric` (str, default='conll') - **Important**
  - Metric for model selection during training
  - Options: `conll`, `lea`, `pd_lea` (phrasal diversity weighted LEA)
  - Recommended: `pd_lea` (best correlates with cross-document performance)

**Data Augmentation:**
- `--additional_sampling` (flag, default=False) - **Recommended**
  - Enables count-based subsampling (Stage 2) to prevent some subtopics to have considerably more samples due to the similarity distribution
  - Reduces training time and prevents overfitting on large topics
  - **Use this flag for best results**

#### Other Parameters
- `--val_batch_size` (int, default=64)
- `--context_method` (str, default='hsu'): Context extraction method (hsu was the only implemented one)
- `--special_tokens` (list, default=['E']): Tokens to mark mentions
- `--job_id` (str, default=''): Optional job ID for logging
- `--world_size` (int, default=1): Number of GPUs for distributed training

### Hardware Requirements

**From Production Training Configuration:**
- **GPU**: NVIDIA A100 (40GB) or equivalent
- **Memory**: 30GB GPU memory minimum
- **CPUs**: 4 cores recommended

**Lower-End Alternative:**
- GPU: NVIDIA V100 (32GB) or RTX 3090
- Reduce batch size if memory issues occur

### Usage Examples

#### Best Configuration (Recommended)
```bash
python train.py \
    --mention_type event \
    --data_setup 1 \
    --language_model bart-large \
    --train_batch_size 32 \
    --val_batch_size 128 \
    --num_epochs 100 \
    --learning_rate 2e-7 \
    --margin 0.3 \
    --clustering_threshold 0.2 \
    --metric pd_lea \
    --additional_sampling
```

#### Quick Test Run (5 epochs)
```bash
python train.py \
    --mention_type event \
    --num_epochs 5 \
    --additional_sampling
```

#### Entity Coreference
```bash
python train.py \
    --mention_type entity \
    --data_setup 0 \
    --num_epochs 100 \
    --additional_sampling
```

### Model Output

Models are saved to:
```
models/{mention_type}/{language_model}/tokens_{special_tokens}/
siamese_model_batch_{train_batch_size}_epochs_{num_epochs}_margin_{margin}_threshold_{clustering_threshold}_setup_{data_setup}.pth
```

Example:
```
models/event/bart-large/tokens_E/siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_1.pth
```

---

## Inference

### Entry Point: `predict.py`

Evaluate a trained model on multiple test datasets.

### Command-Line Arguments

- `--model_path` (str, **required**)
  - Path to trained model `.pth` file
  - Example: `models/event/bart-large/tokens_E/siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_0.pth`

- `--export_predictions` (flag, default=False)
  - Export detailed predictions to CSV files in `predictions/` folder

- `--without_singletons` (flag, default=False) - **Recommended**
  - Exclude singleton clusters from evaluation
  - Standard practice in CDCR evaluation

- `--predicted_topics` (flag, default=False)
  - Use predicted topics instead of gold topics for ECBplus test set
  - Tests model's robustness to topic clustering errors

### Test Datasets

The script automatically evaluates on:
1. **ECBplus test set**: Standard benchmark for event/entity CDCR
2. **NewsWCL50**: News articles with within/cross-document coreference
3. **NP4E**: Event coreference dataset
4. **GVC**: Gun violence corpus

### Hardware Requirements

**From Production Inference Configuration:**
- **GPU**: NVIDIA V100 (16GB) or better
- **Memory**: 30GB GPU memory (handles large test sets)
- **CPUs**: 4 cores
- **Inference Time**: ~20 minutes for all test sets

### Usage Examples

#### Standard Evaluation
```bash
python predict.py \
    --model_path models/event/bart-large/tokens_E/siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_0.pth \
    --without_singletons
```

#### Export Predictions for Analysis
```bash
python predict.py \
    --model_path models/event/bart-large/tokens_E/siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_0.pth \
    --without_singletons \
    --export_predictions
```

#### Test with Predicted Topics
```bash
python predict.py \
    --model_path models/event/bart-large/tokens_E/siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_0.pth \
    --without_singletons \
    --predicted_topics
```

### Output Format

**Console Output:**
```
Processing dataset ecbplus_test
ecbplus_test results: [metrics table]
ecbplus_test test results: CoNLL F1 / LEA F1 / PD_LEA F1: 0.XX/0.XX/0.XX
```

**Exported Predictions** (if `--export_predictions`):
- Files saved to: `predictions/{dataset_name}_{mention_type}.csv`
- Contains: mention IDs, gold clusters, predicted clusters, embeddings

---

## Expected Results

### Performance Benchmarks

**Best Model Configuration:**
- Language Model: BART-large
- Training: ECBplus + FCC-T (data_setup=1)
- Hyperparameters: margin=0.3, threshold=0.2, 100 epochs
- Additional sampling: Enabled

**Expected F1 Scores (without singletons):**

| Dataset | CoNLL F1 | LEA F1 | PD_LEA F1 |
|---------|----------|--------|-----------|
| ECBplus test (events) | ~0.82-0.85 | ~0.79-0.82 | ~0.75-0.78 |
| NewsWCL50 (events) | ~0.70-0.75 | ~0.68-0.72 | ~0.65-0.69 |

**Notes:**
- Results vary by ±2-3 F1 points due to random initialization
- Entity coreference typically achieves 3-5 points lower than events
- Cross-dataset training (data_setup=1,2) improves generalization by ~2-4 F1 points


### File Outputs

After training and prediction:
```
CL_COREF/
├── models/
│   └── event/
│       └── bart-large/
│           └── tokens_E/
│               └── siamese_model_batch_32_epochs_100_margin_0.3_threshold_0.2_setup_1.pth
├── predictions/
│   ├── ecbplus_test_event.csv
│   ├── newswcl50_event.csv
│   ├── np4e_event.csv
│   └── gvc_event.csv
└── slurm_files/
    ├── slurm-train-cross-encoder-gpu-{job_id}.out
    └── slurm-train-cross-encoder-gpu-{job_id}.err
```

---

## Requirements

### Python Dependencies
```
torch>=1.9.0
transformers>=4.18.0
sentence-transformers>=2.2.0
numpy
pandas
nltk
scikit-learn
scipy
```

### Data Prerequisites
- CoNLL-formatted CDCR datasets (ECBplus, FCC-T, NP4E, etc.)
- Preprocessed with mention annotations (entity_mentions.json, event_mentions.json)
- Located in `../CDCR/data/` relative to CL_COREF folder

---

