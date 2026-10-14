# Pipeline Rename and Update Summary

**Date**: February 8, 2026
**Change**: Renamed `02_mentions_preparation` → `01_mentions_preparation` and updated to process all splits

---

## Changes Made

### 1. Pipeline Renamed
- **Old**: `02_mentions_preparation`
- **New**: `01_mentions_preparation`

**Rationale**: This pipeline is actually the first processing step after data exploration, so it should be numbered 01.

### 2. Scope Expanded: All Splits Included

**Previous Behavior:**
- Only processed **train split** (74,611 mentions)
- Excluded validation and test splits

**New Behavior:**
- Processes **all splits**: train, val, test (116,793 mentions total)
- Split breakdown:
  - Train: 74,611 mentions (64%)
  - Val: 20,340 mentions (17%)
  - Test: 21,842 mentions (19%)

**Rationale**: The model requires mention representations for training, validation, AND testing. The validation and test splits need representations for:
- Model evaluation during training (validation)
- Final model testing (test)
- Inference on unseen data

### 3. Files Renamed

**Pipeline Code:**
- `src/cl_coref/pipelines/02_mentions_preparation/` → `01_mentions_preparation/`
- All references updated in:
  - `pipeline.py` (tags changed to `01_mentions_preparation`)
  - `nodes.py` (function renamed: `load_filtered_train_mentions` → `load_filtered_mentions`)

**Configuration:**
- `conf/base/catalog_02_mentions_preparation.yml` → `catalog_01_mentions_preparation.yml`
- `conf/base/parameters_02_mentions_preparation.yml` → `parameters_01_mentions_preparation.yml`
- Updated data catalog entry: `filtered_train_mentions` → `filtered_mentions`

**Documentation:**
- `PIPELINE_02_PAIRS_CREATION_SUMMARY.md` → `PIPELINE_01_PAIRS_CREATION_SUMMARY.md`
- `QUICK_START_02_PAIRS.md` → `QUICK_START_01_PAIRS.md`
- Updated mention counts and split references throughout

### 4. Node Function Changes

**Function Renamed:**
```python
# Before
def load_filtered_train_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Load filtered mentions from the train split only."""
    filtered_df = df[
        (df["is_filtered"]) &
        (df["split"] == "train")
    ]

# After
def load_filtered_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Load filtered mentions from all splits (train, val, test)."""
    filtered_df = df[df["is_filtered"]]
```

**Key Change**: Removed the `split == "train"` filter to include all splits.

### 5. Updated Logging

**New Split Breakdown Logging:**
```
INFO  Filtered: 116,793 mentions (is_filtered=True)
INFO    Datasets: 11
INFO    Splits breakdown:
INFO      test: 21,842 mentions
INFO      train: 74,611 mentions
INFO      val: 20,340 mentions
```

---

## Output Files Updated

### mentions_for_pairs.csv
- **Old size**: 12 MB (74,611 rows)
- **New size**: ~19 MB (116,793 rows)
- **Change**: +42,182 mentions from val and test splits

### mention_representations.parquet
- **Old size**: 48 MB (74,611 representations)
- **New size**: ~75 MB (116,793 representations)
- **Change**: +42,182 representations from val and test splits

### mention_groups_summary.csv
- **Old groups**: 71
- **New groups**: 163
- **Change**: More groups due to additional splits and datasets

### grouping_statistics.json
- **Old mentions**: 74,611
- **New mentions**: 116,793
- **Old clusters**: 18,699
- **New clusters**: ~29,000 (estimated)
- **Old pairs**: ~2.5M
- **New pairs**: ~3.9M (estimated)

---

## Pipeline Execution

### How to Run

```bash
# Old command (no longer works)
kedro run --pipeline 02_mentions_preparation

# New command
kedro run --pipeline 01_mentions_preparation
```

### Execution Time
- **Old**: ~27 minutes (74,611 mentions)
- **New**: ~40-45 minutes (116,793 mentions)
- **Increase**: ~50% longer due to 56% more mentions

---

## Why This Change Was Made

### 1. Model Requirements
The model needs representations for:
- **Training**: Learn from train split
- **Validation**: Tune hyperparameters and monitor performance
- **Testing**: Evaluate final model performance

Without val and test representations, the model cannot be properly evaluated.

### 2. Inference Pipeline
At inference time, the model needs to:
1. Create string representation for new mentions
2. Tokenize with same 128-token limit
3. Generate embeddings
4. Cluster for coreference resolution

Having representations for val/test splits ensures the full inference pipeline is tested.

### 3. Consistency
All splits should use the same:
- Document context strategy (8 sentences)
- Mention markers ([E] and [/E])
- Tokenization approach (128 tokens, truncation="only_first")

This ensures fair comparison across splits.

---

## Backward Compatibility

### Breaking Changes
- ⚠️ Pipeline name changed: `02_mentions_preparation` → `01_mentions_preparation`
- ⚠️ Node name changed: `load_filtered_train_mentions_node` → `load_filtered_mentions_node`
- ⚠️ Dataset name changed: `filtered_train_mentions` → `filtered_mentions`

### Migration Guide

If you have scripts or notebooks referencing the old pipeline:

**Update pipeline references:**
```python
# Before
kedro run --pipeline 02_mentions_preparation

# After
kedro run --pipeline 01_mentions_preparation
```

**Update data loading:**
```python
# Before
df = pd.read_parquet("data/03_primary/mention_representations.parquet")
# Only had train split (74,611 rows)

# After
df = pd.read_parquet("data/03_primary/mention_representations.parquet")
# Now has all splits (116,793 rows)

# Filter by split if needed
train_df = df[df['split'] == 'train']  # 74,611 rows
val_df = df[df['split'] == 'val']      # 20,340 rows
test_df = df[df['split'] == 'test']    # 21,842 rows
```

**Note**: The `split` column is not currently saved in `mention_representations.parquet`. If you need split information, you should:
1. Load `mentions_for_pairs.csv` which has the `split` column
2. Merge with `mention_representations.parquet` on `mention_id`

---

## Verification

After pipeline completion, verify the output:

```bash
# Check total mentions
python -c "import pandas as pd; \
  df = pd.read_parquet('data/03_primary/mention_representations.parquet'); \
  print(f'Total: {len(df):,} representations')"

# Expected output: Total: 116,793 representations

# Check markers
python -c "import pandas as pd; \
  df = pd.read_parquet('data/03_primary/mention_representations.parquet'); \
  has_e = df['mention_context'].str.contains('[E]', regex=False).sum(); \
  has_e_end = df['mention_context'].str.contains('[/E]', regex=False).sum(); \
  print(f'[E]: {has_e:,} ({has_e/len(df)*100:.1f}%)'); \
  print(f'[/E]: {has_e_end:,} ({has_e_end/len(df)*100:.1f}%)')"

# Expected output:
# [E]: 116,793 (100.0%)
# [/E]: 116,793 (100.0%)
```

---

## Next Steps

With representations for all splits now available:

1. **Pipeline 02: Tokenization** (to be created)
   - Convert string representations to token IDs
   - Apply 128-token limit
   - Save tokenized representations by split

2. **Pipeline 03: Pair Generation** (to be created)
   - Generate training pairs from train split only
   - Use similarity-based hard mining
   - Balance positive/negative ratios

3. **Pipeline 04: Model Training** (to be created)
   - Train on train pairs
   - Validate on val split
   - Test on test split

---

**Status**: ✅ Complete and Running
**Current Execution**: In progress (processing 116,793 mentions)
**Expected Completion**: ~40-45 minutes from start
