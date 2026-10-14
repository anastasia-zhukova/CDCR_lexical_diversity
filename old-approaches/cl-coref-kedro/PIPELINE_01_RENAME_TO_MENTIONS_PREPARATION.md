# Pipeline Rename: 01_pairs_creation → 01_mentions_preparation

**Date**: February 9, 2026
**Reason**: The pipeline doesn't create pairs; it prepares mentions for downstream pair selection.

---

## What Changed

### Pipeline Name
- **Before**: `01_pairs_creation`
- **After**: `01_mentions_preparation`

### Why This Name is Better
The pipeline **prepares mentions** by:
1. Loading filtered mentions (all splits)
2. Creating string representations
3. Adding document context (8 sentences)
4. Inserting [E] and [/E] markers

It does **NOT** create pairs. Pair selection happens in pipeline 02.

---

## Files Updated

### Pipeline Code
```
src/cl_coref/pipelines/
└── 01_mentions_preparation/  (renamed from 01_pairs_creation/)
    ├── __init__.py
    ├── pipeline.py  (updated tags and descriptions)
    └── nodes.py     (no changes needed)
```

### Configuration
```
conf/base/
├── catalog_01_mentions_preparation.yml  (renamed from catalog_01_pairs_creation.yml)
└── parameters_01_mentions_preparation.yml  (renamed from parameters_01_pairs_creation.yml)
```

### Documentation
```
├── PIPELINE_01_MENTIONS_PREPARATION_SUMMARY.md  (renamed)
├── QUICK_START_01_MENTIONS_PREPARATION.md  (renamed)
├── README_PIPELINE_01_MENTIONS_PREPARATION.md  (renamed)
└── PIPELINE_RENAME_SUMMARY.md  (updated)
```

---

## How to Use

### Run the Pipeline
```bash
# New command (correct name)
kedro run --pipeline 01_mentions_preparation

# Old command (no longer works)
kedro run --pipeline 01_pairs_creation  # ❌ This will fail
```

### Check Available Pipelines
```bash
kedro registry list

# Output:
# - 00_data_exploration
# - 01_mentions_preparation
# - __default__
```

---

## Pipeline Tags Updated

All node tags changed from `pairs_creation` to `mentions_preparation`:

```python
# Before
tags=["load", "pairs_creation"]

# After
tags=["load", "mentions_preparation"]
```

Pipeline tag also updated:
```python
# Before
tags="01_pairs_creation"

# After
tags="01_mentions_preparation"
```

---

## What This Pipeline Does

**Input**: 175,758 total mentions from 00_data_exploration

**Process**:
1. Filter to mentions with `is_filtered=True` (passed topic filters)
2. Include all splits: train (74,611) + val (20,340) + test (21,842) = 116,793
3. Select 12 required columns
4. Load documents and create string representations:
   - Document context: 8 sentences (excluding mention's sentence)
   - Mention context: Sentence with `[E] mention [/E]` markers
   - Combined: `"{doc_context} [SEP] {mention_context}"`

**Output**:
- `mention_representations.parquet` - 116,793 string representations
- `mentions_for_pairs.csv` - Metadata for all mentions
- `mention_groups_summary.csv` - 163 groups
- `grouping_statistics.json` - Statistics

**Next Pipeline**: 02_pairs_selection (to be created)
- Will use these representations
- Will select hard positive/negative pairs
- Will only use train split (74,611 mentions)

---

## Backward Compatibility

### Breaking Changes
⚠️ Pipeline name changed completely

### Migration Steps

If you have scripts referencing the old name:

**1. Update Kedro commands:**
```bash
# Find and replace in scripts
sed -i 's/01_pairs_creation/01_mentions_preparation/g' your_script.sh
```

**2. Update Python imports (if any):**
```python
# Before
from cl_coref.pipelines.01_pairs_creation import ...

# After
from cl_coref.pipelines.01_mentions_preparation import ...
```

**3. Update configuration references:**
```yaml
# Before
parameters_01_pairs_creation.yml

# After
parameters_01_mentions_preparation.yml
```

---

## Verification

```bash
# 1. Check pipeline is registered
kedro registry list | grep mentions_preparation

# 2. Run pipeline (should complete in ~27 minutes)
kedro run --pipeline 01_mentions_preparation

# 3. Verify output
python -c "import pandas as pd; \
  df = pd.read_parquet('data/03_primary/mention_representations.parquet'); \
  print(f'✓ {len(df):,} representations created')"

# Expected: ✓ 116,793 representations created
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Pipeline name | `01_pairs_creation` | `01_mentions_preparation` |
| What it does | Misleading name | Accurate name |
| Actual function | Prepares mentions | Prepares mentions |
| Pairs created? | No | No |
| When are pairs created? | Never | Pipeline 02 |

**Status**: ✅ Rename complete and verified
**Pipeline**: Ready to use with new name
**Next Step**: Create pipeline 02_pairs_selection
