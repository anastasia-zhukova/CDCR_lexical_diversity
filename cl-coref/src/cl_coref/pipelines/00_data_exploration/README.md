# Data Exploration Pipeline (00_data_exploration)

This pipeline loads all CDCR mention data from multiple datasets and creates a unified dataset for analysis and statistics.

## Overview

The pipeline scans all datasets in the raw data directory and:
1. Finds all `entity_mentions.json` and `event_mentions.json` files
2. Extracts metadata from file paths (dataset_name, split, entity_style)
3. Adds metadata fields to each mention record
4. Combines all mentions into a single unified DataFrame
5. Saves to Parquet format for efficient analysis
6. Computes comprehensive statistics

## Why Parquet?

The unified dataset is saved in **Parquet format** because:

- ✅ **Efficient columnar storage** - Optimized for analytics and filtering
- ✅ **Native support for nested lists** - Stores `tokens_text`, `mention_context` without serialization
- ✅ **Built-in compression** - Typically 10-50x smaller than JSON
- ✅ **Fast filtering** - Column pruning and predicate pushdown
- ✅ **Schema enforcement** - Type safety and validation
- ✅ **Universal compatibility** - Works with pandas, dask, spark, polars
- ✅ **Partitioning support** - Can partition by dataset/split for large datasets

## Pipeline Structure

```
load_all_mentions_node
    ↓
    ├─→ create_summary_table_node
    ├─→ compute_statistics_node ──→ save_statistics_node
    └─→ save_parquet_node
```

## Data Path Structure

Expected directory structure:
```
data/01_raw/cdcr_datasets/
├── ECBplus/
│   ├── train/
│   │   ├── entity_mentions.json
│   │   └── event_mentions.json
│   ├── val/
│   │   ├── entity_mentions.json
│   │   └── event_mentions.json
│   └── test/
│       ├── entity_mentions.json
│       └── event_mentions.json
├── HyperCoref/
│   ├── train/
│   │   ├── entity_mentions.json
│   │   └── event_mentions.json
│   └── ...
└── [other datasets]/
    └── ...
```

## Added Metadata Fields

The pipeline adds three fields to each mention:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `dataset_name` | str | Name of the dataset | "ECBplus" |
| `split` | str | Data split | "train", "val", "test" |
| `entity_style` | str | Mention style | "entity" or "event" |

These fields are extracted from the file path structure:
```
.../dataset_name/split/entity_style_mentions.json
```

## Output Files

### 1. all_mentions.parquet

Unified parquet file with all mentions from all datasets.

**Location**: `data/02_intermediate/all_mentions.parquet`

**Size**: Typically 10-50x smaller than equivalent JSON

**Usage**:
```python
import pandas as pd

# Load all mentions
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

# Filter by dataset
ecb_df = df[df["dataset_name"] == "ECBplus"]

# Filter by split
train_df = df[df["split"] == "train"]

# Filter by entity style
entity_df = df[df["entity_style"] == "entity"]

# Complex queries
hypercoref_train_events = df[
    (df["dataset_name"] == "HyperCoref") &
    (df["split"] == "train") &
    (df["entity_style"] == "event")
]
```

### 2. dataset_statistics.json

Comprehensive statistics about the unified dataset.

**Location**: `data/02_intermediate/dataset_statistics.json`

**Contents**:
```json
{
  "total_mentions": 124798,
  "total_files": 66,
  "by_dataset": {
    "ECBplus": {
      "total_mentions": 8549,
      "entity_mentions": 5731,
      "event_mentions": 2818,
      "splits": ["train", "val", "test"],
      "num_topics": 25
    },
    ...
  },
  "by_split": {
    "train": {
      "total_mentions": 89542,
      "num_datasets": 11
    },
    ...
  },
  "by_entity_style": {
    "entity": 62341,
    "event": 62457
  },
  "cluster_stats": {
    "total_clusters": 24567,
    "singletons": 15432,
    "non_singletons": 9135,
    "avg_cluster_size": 5.08,
    "max_cluster_size": 388,
    "estimated_positive_pairs": 1196845,
    "estimated_negative_pairs": 2393690,
    "estimated_total_pairs": 3590535
  }
}
```

## Running the Pipeline

### From Command Line

```bash
# Run the entire pipeline
kedro run --pipeline 00_data_exploration

# Run only loading and statistics (skip saving)
kedro run --pipeline 00_data_exploration --tags load,statistics

# Run only loading and saving (skip statistics)
kedro run --pipeline 00_data_exploration --tags load,save
```

### From Python

```python
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Bootstrap project
bootstrap_project("/path/to/cl-coref")

# Run pipeline
with KedroSession.create() as session:
    session.run(pipeline_name="00_data_exploration")
```

## Parameters

Configure in `conf/base/parameters_00_data_exploration.yml`:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data_path` | str | Path to raw datasets | "data/01_raw/cdcr_datasets" |
| `output_parquet_path` | str | Where to save parquet | "data/02_intermediate/all_mentions.parquet" |
| `output_stats_path` | str | Where to save stats | "data/02_intermediate/dataset_statistics.json" |
| `partition_cols` | list | Partitioning columns | null (single file) |

### Partitioning Options

For very large datasets (>100M rows), you can enable partitioning:

```yaml
# Single file (recommended for most cases)
partition_cols: null

# Partition by dataset (creates subdirectories per dataset)
partition_cols: ["dataset_name"]

# Partition by dataset and split
partition_cols: ["dataset_name", "split"]
```

**Partitioned output structure**:
```
all_mentions.parquet/
├── dataset_name=ECBplus/
│   ├── split=train/
│   │   └── part-0.parquet
│   ├── split=val/
│   │   └── part-0.parquet
│   └── split=test/
│       └── part-0.parquet
├── dataset_name=HyperCoref/
│   └── ...
└── ...
```

## Node Functions

All nodes are defined in `nodes.py`:

### Core Functions

- **`load_all_mentions_from_datasets()`** - Recursively loads all mention JSON files and adds metadata
- **`create_summary_table()`** - Creates a summary table with counts per dataset/split/entity_style
- **`compute_dataset_statistics()`** - Computes comprehensive statistics including cluster analysis
- **`save_mentions_to_parquet()`** - Saves unified DataFrame to Parquet with optional partitioning
- **`save_statistics_to_json()`** - Saves statistics dictionary to JSON

## Example Analysis

After running the pipeline, you can analyze the data:

```python
import pandas as pd
import json

# Load unified data
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

# Load statistics
with open("data/02_intermediate/dataset_statistics.json") as f:
    stats = json.load(f)

# What datasets do we have?
print(df["dataset_name"].value_counts())

# How many mentions per split?
print(df.groupby("split").size())

# Entity vs event distribution per dataset
print(df.groupby(["dataset_name", "entity_style"]).size())

# How many topics per dataset?
print(df.groupby("dataset_name")["topic"].nunique())

# What are the topics in HyperCoref?
hypercoref_topics = df[df["dataset_name"] == "HyperCoref"]["topic"].unique()
print(sorted(hypercoref_topics))

# Cluster size distribution
cluster_sizes = df.groupby(["dataset_name", "split", "coref_chain"]).size()
print(cluster_sizes.describe())

# Largest clusters
print(cluster_sizes.nlargest(10))
```

## Memory Efficiency

The pipeline is designed to be memory-efficient:

1. **Streaming loading**: Loads one JSON file at a time
2. **Efficient storage**: Parquet uses columnar compression
3. **Lazy evaluation**: Parquet supports reading subsets of columns/rows

For very large datasets, consider:
- Using partitioning to break data into smaller chunks
- Using dask or spark for out-of-core processing
- Reading only required columns: `pd.read_parquet(path, columns=["mention_id", "topic"])`

## Integration with Analysis Tools

The parquet format integrates with many tools:

**Pandas**:
```python
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")
```

**Dask** (for larger-than-memory data):
```python
import dask.dataframe as dd
df = dd.read_parquet("data/02_intermediate/all_mentions.parquet")
```

**PyArrow**:
```python
import pyarrow.parquet as pq
table = pq.read_table("data/02_intermediate/all_mentions.parquet")
```

**Polars** (faster alternative to pandas):
```python
import polars as pl
df = pl.read_parquet("data/02_intermediate/all_mentions.parquet")
```

**DuckDB** (SQL queries on parquet):
```python
import duckdb
duckdb.sql("SELECT dataset_name, COUNT(*) FROM 'data/02_intermediate/all_mentions.parquet' GROUP BY dataset_name")
```

## See Also

- Original analysis script: `analyze_datasets_lightweight.py`
- Example notebook: `load_json_example.ipynb`
- Configuration: `conf/base/parameters_00_data_exploration.yml`
