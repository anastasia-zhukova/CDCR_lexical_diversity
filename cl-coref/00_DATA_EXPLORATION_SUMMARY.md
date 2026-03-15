# Pipeline 00: Data Exploration - Quick Reference

## What It Does

Loads ALL mention data from ALL datasets and creates:
1. **Unified Parquet file** - All mentions in one efficient file
2. **Statistics JSON** - Comprehensive analysis with cluster and pair estimates
3. **Summary CSV** - Quick reference table

## Why Parquet?

✅ **10-50x smaller than JSON**
✅ **Native support for nested lists** (tokens_text, mention_context)
✅ **Fast filtering** - Read only needed columns/rows
✅ **Universal compatibility** - pandas, dask, spark, polars, DuckDB

## What Gets Added

The pipeline adds 3 metadata fields extracted from file paths:

| Field | Source | Example |
|-------|--------|---------|
| `dataset_name` | Grandparent directory | "ECBplus" |
| `split` | Parent directory | "train" |
| `entity_style` | Filename | "entity" or "event" |

**Path structure**: `data/01_raw/cdcr_datasets/{dataset_name}/{split}/{entity_style}_mentions.json`

## Files Created

```
data/02_intermediate/
├── all_mentions.parquet          (~15 MB) - Unified mentions
├── dataset_statistics.json       (~5 KB)  - Comprehensive stats
└── summary_table.csv             (~2 KB)  - Quick reference
```

## Quick Start

### 1. Run the pipeline
```bash
cd cl-coref
kedro run --pipeline 00_data_exploration
```

### 2. Load the data
```python
import pandas as pd

# Load all mentions
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

# Filter examples
ecb_train = df[(df["dataset_name"] == "ECBplus") & (df["split"] == "train")]
all_events = df[df["entity_style"] == "event"]
hypercoref_topics = df[df["dataset_name"] == "HyperCoref"]["topic"].unique()
```

### 3. View statistics
```python
import json

with open("data/02_intermediate/dataset_statistics.json") as f:
    stats = json.load(f)

print(f"Total mentions: {stats['total_mentions']:,}")
print(f"Estimated pairs: ~{stats['cluster_stats']['estimated_total_pairs']:,}")
```

## Expected Output

```
Total mentions: ~125,000
Datasets: 11
Splits: train, val, test
Entity/Event: ~50/50 split
Clusters: ~25,000
Estimated pairs: ~3.6M
```

## Testing

Test the pipeline without Kedro:
```bash
cd cl-coref
python test_data_exploration_pipeline.py
```

## File Locations

```
cl-coref/
├── src/cl_coref/pipelines/00_data_exploration/
│   ├── nodes.py                 # 5 functional nodes
│   ├── pipeline.py              # Pipeline definition
│   └── README.md                # Full documentation
├── conf/base/
│   ├── parameters_00_data_exploration.yml
│   └── catalog_00_data_exploration.yml
├── test_data_exploration_pipeline.py
└── 00_DATA_EXPLORATION_SUMMARY.md (this file)
```

## Key Design Decisions

1. **Single file, not partitioned** - Simpler and faster for ~125K mentions
2. **Parquet, not JSON** - 10-50x smaller, much faster
3. **Metadata extraction** - Automatic from file paths
4. **Pure functions** - Easy to test and reuse

## Next Steps

After running this pipeline:
1. **Explore the data** - Use pandas/jupyter to analyze
2. **Build filtering pipeline** - Apply topic filters (e.g., HyperCoref: us/politics/international)
3. **Generate training pairs** - Create actual positive/negative pairs
4. **Train models** - Use filtered data for CDCR training

## Troubleshooting

**Pipeline fails with "data path not found":**
- Check that `data/01_raw/cdcr_datasets/` exists
- Verify it contains dataset folders (ECBplus, HyperCoref, etc.)

**Parquet file is missing:**
- Check `data/02_intermediate/` directory exists
- Look for errors in pipeline logs

**Statistics are wrong:**
- Verify JSON files have `coref_chain` field
- Check that mentions have proper structure

## Performance

**Loading time**: ~30 seconds (first run)
**Parquet read time**: ~2 seconds (subsequent reads)
**Memory usage**: ~500 MB during loading, ~200 MB for DataFrame
**Disk usage**: ~15 MB (parquet) vs ~150 MB (JSON)

**Compression ratio**: 10x (snappy compression)

## Integration

This pipeline is the **foundation** for all downstream analysis:

```
00_data_exploration (this pipeline)
    ↓
    ├─→ 01_dataset_filtering (apply topic filters)
    ├─→ 02_statistics_analysis (detailed per-dataset analysis)
    ├─→ 03_pair_generation (create training pairs)
    └─→ 04_model_training (load filtered data for training)
```

## See Also

- **Full documentation**: `src/cl_coref/pipelines/00_data_exploration/README.md`
- **Implementation details**: `PIPELINE_00_DATA_EXPLORATION.md`
- **Test script**: `test_data_exploration_pipeline.py`
- **Configuration**: `conf/base/parameters_00_data_exploration.yml`
