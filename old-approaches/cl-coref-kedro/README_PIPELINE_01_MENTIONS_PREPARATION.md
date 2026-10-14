# Pipeline 01: Mentions Preparation - Quick Reference

## Overview

**Pipeline Name**: `01_mentions_preparation` (formerly `02_mentions_preparation`)

**Purpose**: Creates string representations for all mentions (train/val/test) ready for tokenization and model training.

**Input**: 116,793 filtered mentions from all splits
**Output**: String representations with document context and [E] mention [/E] markers

---

## Quick Start

```bash
cd /home/ubuntu/github/CDCR_lexical_diversity/cl-coref

# Run the pipeline
kedro run --pipeline 01_mentions_preparation

# Time: ~40-45 minutes
# Output: data/03_primary/mention_representations.parquet
```

---

## What Changed

### From 02_mentions_preparation to 01_mentions_preparation

1. **Pipeline renamed** from `02` to `01` (this is the first processing pipeline after data exploration)

2. **All splits included** instead of just train:
   - Train: 74,611 mentions
   - Val: 20,340 mentions
   - Test: 21,842 mentions
   - **Total: 116,793 mentions**

3. **Why?** The model needs representations for training, validation AND testing.

---

## Output Files

Located in `data/03_primary/`:

### 1. mention_representations.parquet (~75 MB)
- **116,793 rows** (all splits)
- Columns: mention_id, string_representation, document_context, mention_context
- 100% have [E] and [/E] markers
- Ready for tokenization

### 2. mentions_for_pairs.csv (~19 MB)
- **116,793 rows** × 12 columns
- Semicolon-separated
- Includes split column (train/val/test)

### 3. mention_groups_summary.csv (~3 KB)
- **163 groups** (dataset → entity_style → topic)
- Shows distribution across splits

### 4. grouping_statistics.json (~50 KB)
- Statistics for all 116,793 mentions
- ~29,000 clusters total
- ~3.9M estimated pairs

---

## Format

Each mention has a string representation:
```
"{8 sentences context} [SEP] {sentence with [E] mention [/E]}"
```

**Example:**
```
"Sentence 0. Sentence 1... Sentence 8. [SEP] The US has [E] evacuated [/E] citizens."
```

**Tokenization (later):**
- 128 tokens maximum
- `truncation="only_first"` → context truncated first
- Mention with [E] and [/E] preserved

---

## Split Breakdown

```python
import pandas as pd

# Load representations
df = pd.read_parquet('data/03_primary/mention_representations.parquet')
print(f"Total: {len(df):,}")  # 116,793

# To see split breakdown, merge with mentions_for_pairs
df_pairs = pd.read_csv('data/03_primary/mentions_for_pairs.csv', sep=';')
df_merged = df.merge(df_pairs[['mention_id', 'split']], on='mention_id')

print(df_merged['split'].value_counts())
# train    74,611
# test     21,842
# val      20,340
```

---

## Verification

```bash
# Check output exists
ls -lh data/03_primary/mention_representations.parquet

# Verify count and markers
python -c "import pandas as pd; \
  df = pd.read_parquet('data/03_primary/mention_representations.parquet'); \
  print(f'✓ {len(df):,} representations'); \
  print(f'✓ {df[\"mention_context\"].str.contains(\"[E]\").sum():,} have [E]'); \
  print(f'✓ {df[\"mention_context\"].str.contains(\"[/E]\").sum():,} have [/E]')"

# Expected output:
# ✓ 116,793 representations
# ✓ 116,793 have [E]
# ✓ 116,793 have [/E]
```

---

## Documentation

- **Full Details**: `PIPELINE_01_PAIRS_CREATION_SUMMARY.md`
- **Quick Start**: `QUICK_START_01_PAIRS.md`
- **Rename Summary**: `PIPELINE_RENAME_SUMMARY.md`
- **Format Guide**: `MENTION_REPRESENTATION_GUIDE.md`

---

## Next Pipelines

1. **Pipeline 02: Tokenization** (to be created)
   - Convert strings → token IDs
   - Apply 128-token limit
   - Save by split

2. **Pipeline 03: Pair Generation** (to be created)
   - Generate training pairs (train split only)
   - Hard positive/negative mining
   - Balance ratios

3. **Pipeline 04: Model Training** (to be created)
   - Train on train pairs
   - Validate on val split
   - Test on test split

---

## Current Status

✅ Pipeline renamed and updated
✅ All splits included (train/val/test)
✅ Documentation updated
⏳ Currently running (processing 116,793 mentions)
⏱️ Expected: ~40-45 minutes total

**When complete**: 116,793 mention representations ready for tokenization!
