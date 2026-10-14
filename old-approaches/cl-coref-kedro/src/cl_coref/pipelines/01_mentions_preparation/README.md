# Pipeline 02: Pairs Creation

## Overview

This pipeline prepares filtered mentions for pair generation by organizing them into hierarchical groups: **dataset → entity_style → topic**.

## What It Does

1. **Loads filtered train mentions** - Only uses mentions where:
   - `is_filtered=True` (passed topic filters from pipeline 00)
   - `split='train'` (only training data)

2. **Selects required columns** - Keeps only necessary fields:
   - `coref_chain` - Coreference cluster ID
   - `mention_id` - Unique mention identifier
   - `is_singleton` - Singleton flag
   - `conll_doc_key` - Document key
   - `topic_id` - Numeric topic ID
   - `subtopic_id` - Subtopic ID
   - `dataset_name` - Dataset name
   - `split` - Data split (always 'train')
   - `entity_style` - 'entity' or 'event'
   - `topic` - Topic name/description

3. **Computes grouping statistics** - Creates hierarchical stats:
   ```
   Dataset
   └── Entity Style (entity/event)
       └── Topic
           ├── Mentions count
           ├── Clusters count
           ├── Singletons count
           └── Estimated pairs
   ```

4. **Creates summary table** - CSV with one row per (dataset, entity_style, topic) group

## Pipeline Flow

```
all_mentions_df (from pipeline 00 - Parquet with nested lists)
    ↓
load_filtered_train_mentions_node
    ↓ (filter: is_filtered=True & split='train')
filtered_train_mentions
    ↓
select_required_columns_node
    ↓ (keep only 10 columns - no nested lists)
mentions_for_pairs.csv (semicolon-separated)
    ↓
    ├─→ compute_grouping_statistics_node → grouping_statistics.json
    └─→ create_mention_groups_node → mention_groups_summary.csv
```

## Outputs

### 1. mentions_for_pairs.csv

CSV file (semicolon-separated) with only train mentions that passed filters, keeping only required columns.

**Location**: `data/03_primary/mentions_for_pairs.csv`

**Format**: Semicolon-separated (`;`) CSV - chosen because:
- No nested lists or complex data structures (unlike all_mentions)
- Easy to inspect and work with
- Smaller and simpler than Parquet for this structured data

**Columns** (10 total):
- `coref_chain`
- `mention_id`
- `is_singleton`
- `conll_doc_key`
- `topic_id`
- `subtopic_id`
- `dataset_name`
- `split` (always 'train')
- `entity_style`
- `topic`

**Size**: Much smaller than all_mentions.parquet (only train, only filtered, fewer columns, no nested lists)

### 2. grouping_statistics.json

Hierarchical statistics grouped by dataset → entity_style → topic.

**Location**: `data/03_primary/grouping_statistics.json`

**Structure**:
```json
{
  "total_mentions": 50000,
  "total_datasets": 11,
  "total_splits": ["train"],
  "total_clusters": 15000,
  "total_estimated_pairs": 500000,
  "by_dataset": {
    "HyperCoref": {
      "total_mentions": 12000,
      "total_clusters": 3000,
      "total_pairs": 80000,
      "entity_styles": {
        "event": {
          "total_mentions": 12000,
          "num_topics": 3,
          "total_clusters": 3000,
          "estimated_total_pairs": 80000,
          "topics": {
            "us": {
              "mentions": 5000,
              "clusters": 1200,
              "singletons": 800,
              "non_singletons": 400,
              "avg_cluster_size": 4.2,
              "max_cluster_size": 85,
              "estimated_positive_pairs": 15000,
              "estimated_negative_pairs": 30000,
              "estimated_total_pairs": 45000
            },
            "politics": { ... },
            "international": { ... }
          }
        }
      }
    },
    "WECEng": { ... }
  }
}
```

### 3. mention_groups_summary.csv

Summary table with one row per group.

**Location**: `data/03_primary/mention_groups_summary.csv`

**Columns**:
- `dataset_name`
- `entity_style`
- `topic`
- `mentions`
- `clusters`
- `singletons`
- `non_singletons`

**Example**:
```csv
dataset_name,entity_style,topic,mentions,clusters,singletons,non_singletons
HyperCoref,event,us,5000,1200,800,400
HyperCoref,event,politics,4000,1000,650,350
HyperCoref,event,international,3000,800,500,300
WECEng,event,Earthquake,2500,600,400,200
...
```

## Running the Pipeline

### Command Line
```bash
kedro run --pipeline 02_pairs_creation
```

### From Notebook
```python
%reload_kedro ./
session.run(pipeline_name="02_pairs_creation")
```

### Run Both Pipelines
```bash
# Pipeline 00 first (loads and filters all data)
kedro run --pipeline 00_data_exploration

# Then pipeline 02 (organizes train mentions)
kedro run --pipeline 02_pairs_creation
```

## Usage Examples

### Load Mentions for Pairs
```python
import pandas as pd

# Load mentions ready for pair generation
mentions = pd.read_csv("data/03_primary/mentions_for_pairs.csv", sep=";")

print(f"Total mentions: {len(mentions):,}")
print(f"Datasets: {mentions['dataset_name'].unique()}")
print(f"Entity styles: {mentions['entity_style'].value_counts()}")
print(f"Splits: {mentions['split'].unique()}")  # Should only be ['train']
```

### View Grouping Statistics
```python
import json

with open("data/03_primary/grouping_statistics.json") as f:
    stats = json.load(f)

print(f"Total mentions: {stats['total_mentions']:,}")
print(f"Total pairs: ~{stats['total_estimated_pairs']:,}")

# View specific dataset
hypercoref = stats['by_dataset']['HyperCoref']
print(f"\nHyperCoref:")
for entity_style, style_stats in hypercoref['entity_styles'].items():
    print(f"  {entity_style}: {style_stats['total_mentions']:,} mentions")
    for topic, topic_stats in style_stats['topics'].items():
        print(f"    {topic}: {topic_stats['mentions']:,} mentions, "
              f"~{topic_stats['estimated_total_pairs']:,} pairs")
```

### Work with Specific Groups
```python
# Load summary
summary = pd.read_csv("data/03_primary/mention_groups_summary.csv")

# Sort by largest groups
largest = summary.sort_values("mentions", ascending=False).head(10)
print("Largest groups:")
print(largest)

# Get HyperCoref event groups
hypercoref_events = summary[
    (summary['dataset_name'] == 'HyperCoref') &
    (summary['entity_style'] == 'event')
]
print("\nHyperCoref event topics:")
print(hypercoref_events)

# Filter mentions for a specific group
mentions = pd.read_csv("data/03_primary/mentions_for_pairs.csv", sep=";")
us_events = mentions[
    (mentions['dataset_name'] == 'HyperCoref') &
    (mentions['entity_style'] == 'event') &
    (mentions['topic'] == 'us')
]
print(f"\nHyperCoref 'us' events: {len(us_events):,} mentions")
```

## Validation Checks

The pipeline performs several validation checks:

1. **Split validation**: Ensures only 'train' split is present
   ```
   Expected only 'train' split, but found: ['train', 'val']  # Would warn
   ```

2. **Column validation**: Checks all required columns exist
   ```
   Missing columns: ['coref_chain', 'topic']  # Would error
   ```

3. **Filter validation**: Ensures is_filtered column exists
   ```
   Column 'is_filtered' not found - run 00_data_exploration pipeline first
   ```

## Integration with Pipeline 00

This pipeline depends on `00_data_exploration`:

```
Pipeline 00: Data Exploration
    ↓ (creates all_mentions.parquet with is_filtered column)
Pipeline 02: Pairs Creation
    ↓ (uses only train + filtered mentions)
[Future] Pipeline 03: Pair Generation
```

**Data flow**:
- Pipeline 00 creates: `data/02_intermediate/all_mentions.parquet` (all splits, all mentions, with is_filtered, nested lists)
- Pipeline 02 filters to: `data/03_primary/mentions_for_pairs.csv` (train only, filtered only, reduced columns, no nested lists, CSV format)

## Key Features

### 1. Only Train Data
```python
# Automatically filtered
df['split'].unique()
# Output: ['train']
```

### 2. Only Filtered Mentions
```python
# All mentions passed topic filters
df['is_filtered'].all()  # Would be True if column existed
# (is_filtered column not saved in output to save space)
```

### 3. Hierarchical Grouping
```
Dataset (11 datasets)
└── Entity Style (entity, event)
    └── Topic (varies by dataset)
```

### 4. Reduced Size
```python
# Comparison
all_mentions = pd.read_parquet("data/02_intermediate/all_mentions.parquet")
pairs_mentions = pd.read_csv("data/03_primary/mentions_for_pairs.csv", sep=";")

print(f"All mentions: {len(all_mentions):,} rows, {len(all_mentions.columns)} columns")
print(f"Pairs mentions: {len(pairs_mentions):,} rows, {len(pairs_mentions.columns)} columns")
# Output:
# All mentions: 204,068 rows, 31 columns
# Pairs mentions: 50,000 rows, 10 columns (example numbers)
```

## Next Steps

This pipeline prepares the data for:
- **Pipeline 03**: Actual pair generation (positive and negative pairs)
- **Pipeline 04**: Pair sampling and balancing
- **Pipeline 05**: Training data creation

The hierarchical grouping enables:
- Per-topic pair generation
- Per-dataset pair balancing
- Entity vs event separate processing
