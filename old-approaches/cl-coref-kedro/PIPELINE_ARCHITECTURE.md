# Kedro Pipeline Architecture

## Directory Structure

```
cl-coref/
├── src/cl_coref/pipelines/
│   └── 00_data_exploration/           # Pipeline: Load and unify all datasets
│       ├── __init__.py
│       ├── nodes.py                   # 5 functional nodes
│       ├── pipeline.py                # Pipeline definition
│       └── README.md                  # Full documentation
│
├── conf/base/
│   ├── parameters_00_data_exploration.yml  # Configuration
│   └── catalog_00_data_exploration.yml     # Data catalog
│
├── data/
│   ├── 01_raw/
│   │   └── cdcr_datasets/             # INPUT: Raw JSON files
│   │       ├── ECBplus/
│   │       │   ├── train/
│   │       │   │   ├── entity_mentions.json
│   │       │   │   └── event_mentions.json
│   │       │   ├── val/...
│   │       │   └── test/...
│   │       ├── HyperCoref/...
│   │       └── [9 more datasets]/...
│   │
│   └── 02_intermediate/               # OUTPUT: Unified data
│       ├── all_mentions.parquet       # Unified mentions (15 MB)
│       ├── dataset_statistics.json    # Statistics (5 KB)
│       └── summary_table.csv          # Summary (2 KB)
│
├── test_data_exploration_pipeline.py  # Test script
├── 00_DATA_EXPLORATION_SUMMARY.md     # Quick reference
└── PIPELINE_00_DATA_EXPLORATION.md    # Detailed implementation doc
```

## Pipeline 00: Data Exploration

### Overview

```
INPUT: 66 JSON files (~150 MB)
    ↓
load_all_mentions_from_datasets()
    ↓
DataFrame (~125K rows, 30 columns)
    ↓
    ├─→ create_summary_table() ──→ summary_table.csv
    ├─→ compute_dataset_statistics() ──→ save_statistics_to_json() ──→ dataset_statistics.json
    └─→ save_mentions_to_parquet() ──→ all_mentions.parquet
    ↓
OUTPUT: 3 files (~20 MB total)
```

### Data Flow

```
Raw JSON Files (01_raw/cdcr_datasets/)
    ├── ECBplus/train/entity_mentions.json      [5,731 mentions]
    ├── ECBplus/train/event_mentions.json       [2,818 mentions]
    ├── HyperCoref/train/entity_mentions.json   [0 mentions]
    ├── HyperCoref/train/event_mentions.json    [49,911 mentions]
    └── ... (62 more files)

            ↓ load_all_mentions_from_datasets()

Unified DataFrame
    - All mentions from all files
    - Added metadata: dataset_name, split, entity_style
    - Shape: (~125K rows, 30 columns)
    - Memory: ~500 MB

            ↓ Three parallel operations:

    ┌───────────────┼───────────────┐
    │               │               │
    ↓               ↓               ↓
Summary Table   Statistics    Parquet File
(CSV 2 KB)      (JSON 5 KB)   (15 MB)
```

### Node Functions

```python
# Node 1: Load all mentions and add metadata
def load_all_mentions_from_datasets(data_path: str) -> pd.DataFrame:
    """
    - Scans for *_mentions.json files recursively
    - Extracts dataset_name, split, entity_style from paths
    - Adds metadata to each mention
    - Returns unified DataFrame
    """

# Node 2: Create summary table
def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Groups by dataset_name, split, entity_style
    - Counts mentions and clusters
    - Returns summary DataFrame
    """

# Node 3: Compute statistics
def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    - Stats by dataset, split, entity_style
    - Cluster analysis (singletons, sizes, pairs)
    - Returns statistics dictionary
    """

# Node 4: Save to Parquet
def save_mentions_to_parquet(df, output_path, partition_cols) -> str:
    """
    - Saves DataFrame with snappy compression
    - Optional partitioning (disabled by default)
    - Returns file path
    """

# Node 5: Save statistics to JSON
def save_statistics_to_json(stats, output_path) -> str:
    """
    - Saves statistics dictionary
    - Returns file path
    """
```

## Data Schema

### Input JSON Structure

```json
[
  {
    "coref_chain": "TIM16236441119149059",
    "mention_id": "1_10ecb_TIM16236441119149059_3_11_m433",
    "tokens_str": "on Tuesday",
    "tokens_text": ["on", "Tuesday"],
    "tokens_number": [11, 12],
    "mention_context": ["Perennial", "party", "girl", ...],
    "mention_head": "on",
    "mention_type": "TIM",
    "topic_id": "1",
    "topic": "1",
    "subtopic_id": "1ecb",
    "doc_id": "10",
    "is_singleton": false,
    ...
  }
]
```

### Output Parquet Schema

```
All original fields +
├── dataset_name: string (e.g., "ECBplus")
├── split: string (e.g., "train", "val", "test")
└── entity_style: string (e.g., "entity" or "event")
```

### Output Statistics Structure

```json
{
  "total_mentions": 125000,
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
    "train": {"total_mentions": 89542, "num_datasets": 11},
    "val": {...},
    "test": {...}
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

## Configuration

### parameters_00_data_exploration.yml

```yaml
# Where to find raw JSON files
data_path: "data/01_raw/cdcr_datasets"

# Where to save unified parquet
output_parquet_path: "data/02_intermediate/all_mentions.parquet"

# Where to save statistics
output_stats_path: "data/02_intermediate/dataset_statistics.json"

# Partitioning (null = single file)
partition_cols: null
```

### catalog_00_data_exploration.yml

```yaml
# Unified mentions (Parquet)
all_mentions_df:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/all_mentions.parquet

# Summary table (CSV)
summary_table:
  type: pandas.CSVDataSet
  filepath: data/02_intermediate/summary_table.csv

# Statistics (JSON)
dataset_statistics:
  type: json.JSONDataSet
  filepath: data/02_intermediate/dataset_statistics.json
```

## Usage Examples

### Run the pipeline
```bash
kedro run --pipeline 00_data_exploration
```

### Load and filter data
```python
import pandas as pd

# Load unified data
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

# Filter by dataset
ecbplus = df[df["dataset_name"] == "ECBplus"]

# Filter by split
train = df[df["split"] == "train"]

# Filter by entity style
events = df[df["entity_style"] == "event"]

# Complex filter
hypercoref_train_events = df[
    (df["dataset_name"] == "HyperCoref") &
    (df["split"] == "train") &
    (df["entity_style"] == "event")
]

# Analyze topics
topics_per_dataset = df.groupby("dataset_name")["topic"].nunique()
```

### Query with DuckDB (SQL on Parquet)
```python
import duckdb

# SQL query on parquet file
result = duckdb.sql("""
    SELECT
        dataset_name,
        split,
        entity_style,
        COUNT(*) as mention_count
    FROM 'data/02_intermediate/all_mentions.parquet'
    GROUP BY dataset_name, split, entity_style
    ORDER BY mention_count DESC
""").df()
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Input size** | ~150 MB (JSON) |
| **Output size** | ~15 MB (Parquet) |
| **Compression ratio** | 10x |
| **Loading time** | ~30 seconds (first run) |
| **Read time** | ~2 seconds (Parquet) |
| **Memory usage** | ~500 MB (during load) |
| **Mentions processed** | ~125,000 |
| **Files processed** | 66 |

## Future Pipelines

This pipeline creates the foundation for:

### 01_dataset_filtering
- Apply topic filters (e.g., HyperCoref: us/politics/international)
- Create filtered training datasets
- Balance dataset sizes

### 02_statistics_analysis
- Detailed per-dataset analysis
- Topic distribution analysis
- Cluster size analysis
- Lexical diversity metrics

### 03_pair_generation
- Generate actual positive/negative pairs
- Balance positive/negative ratios
- Create training batches

### 04_model_training
- Load filtered data
- Create data loaders
- Train CDCR models

## Design Principles

1. **Functional nodes** - Pure functions, easy to test
2. **Single responsibility** - Each node does one thing
3. **Efficient storage** - Parquet for analytics
4. **Metadata enrichment** - Add context from paths
5. **Comprehensive stats** - Understand the data
6. **Reusable outputs** - Foundation for downstream pipelines

## Benefits

✅ **Load once, analyze many times**
✅ **10x smaller files** (Parquet compression)
✅ **Fast filtering** (columnar storage)
✅ **Universal compatibility** (pandas, spark, duckdb)
✅ **Automatic metadata** (extracted from paths)
✅ **Comprehensive statistics** (cluster and pair analysis)
