# Pipeline 02: Pairs Creation - Quick Summary

## What It Does

Prepares filtered train mentions for pair generation by organizing them into hierarchical groups:
**Dataset → Entity Style → Topic**

## Key Features

✅ **Only train split** - Filters to `split='train'` only
✅ **Only filtered mentions** - Uses `is_filtered=True` from pipeline 00
✅ **Reduced columns** - Keeps only 10 required columns (vs 30+ in original)
✅ **Hierarchical grouping** - Organized by dataset, entity_style, topic
✅ **Comprehensive statistics** - Cluster counts and pair estimates per group

## Pipeline Flow

```
all_mentions.parquet (from pipeline 00)
    ↓ Filter: is_filtered=True & split='train'
filtered_train_mentions
    ↓ Select 10 required columns
mentions_for_pairs.parquet
    ↓
    ├─→ grouping_statistics.json (hierarchical stats)
    └─→ mention_groups_summary.csv (flat table)
```

## Required Columns (10 total)

The pipeline keeps only these columns:
- `coref_chain` - Cluster ID
- `mention_id` - Unique ID
- `is_singleton` - Singleton flag
- `conll_doc_key` - Document key
- `topic_id` - Numeric topic ID
- `subtopic_id` - Subtopic ID
- `dataset_name` - Dataset name
- `split` - Always 'train'
- `entity_style` - 'entity' or 'event'
- `topic` - Topic name

## Outputs

### 1. mentions_for_pairs.csv
Train mentions with only required columns (semicolon-separated).
- **Location**: `data/03_primary/mentions_for_pairs.csv`
- **Format**: Semicolon-separated CSV (no nested lists, simple tabular data)
- **Size**: Much smaller than all_mentions.parquet
- **Usage**: Foundation for pair generation

### 2. grouping_statistics.json
Hierarchical statistics grouped by dataset → entity_style → topic.
- **Location**: `data/03_primary/grouping_statistics.json`
- **Contains**: Mention counts, cluster stats, pair estimates per group

### 3. mention_groups_summary.csv
Flat table with one row per (dataset, entity_style, topic) group.
- **Location**: `data/03_primary/mention_groups_summary.csv`
- **Usage**: Quick reference for group sizes

## Running the Pipeline

```bash
# From command line
kedro run --pipeline 02_pairs_creation

# Or from notebook
session.run(pipeline_name="02_pairs_creation")
```

**Prerequisites**: Must run pipeline 00 first to create `all_mentions.parquet` with `is_filtered` column.

## Example Usage

### Load Mentions for Pairs
```python
import pandas as pd

mentions = pd.read_csv("data/03_primary/mentions_for_pairs.csv", sep=";")
print(f"Total mentions: {len(mentions):,}")
print(f"Splits: {mentions['split'].unique()}")  # Should be ['train']
```

### View Statistics
```python
import json

with open("data/03_primary/grouping_statistics.json") as f:
    stats = json.load(f)

print(f"Total mentions: {stats['total_mentions']:,}")
print(f"Estimated pairs: ~{stats['total_estimated_pairs']:,}")

# View specific dataset
hypercoref = stats['by_dataset']['HyperCoref']
for entity_style, style_stats in hypercoref['entity_styles'].items():
    print(f"{entity_style}: {style_stats['num_topics']} topics")
```

### Work with Specific Groups
```python
# Load summary
summary = pd.read_csv("data/03_primary/mention_groups_summary.csv")

# Get HyperCoref event topics
hypercoref_events = summary[
    (summary['dataset_name'] == 'HyperCoref') &
    (summary['entity_style'] == 'event')
]
print(hypercoref_events)

# Load mentions for a specific group
mentions = pd.read_csv("data/03_primary/mentions_for_pairs.csv", sep=";")
us_events = mentions[
    (mentions['dataset_name'] == 'HyperCoref') &
    (mentions['entity_style'] == 'event') &
    (mentions['topic'] == 'us')
]
print(f"HyperCoref 'us' events: {len(us_events):,} mentions")
```

## Validation

The pipeline validates:
1. ✓ Only 'train' split present
2. ✓ All required columns exist
3. ✓ `is_filtered` column exists

## Statistics Structure

### Top Level
```json
{
  "total_mentions": 50000,
  "total_splits": ["train"],
  "total_clusters": 15000,
  "total_estimated_pairs": 500000
}
```

### By Dataset
```json
"HyperCoref": {
  "total_mentions": 12000,
  "total_clusters": 3000,
  "total_pairs": 80000,
  "entity_styles": {
    "event": {
      "num_topics": 3,
      "topics": {
        "us": {
          "mentions": 5000,
          "clusters": 1200,
          "estimated_total_pairs": 45000
        }
      }
    }
  }
}
```

## Integration

```
Pipeline 00: Data Exploration
    ↓ (all_mentions.parquet with is_filtered - has nested lists)
Pipeline 02: Pairs Creation
    ↓ (mentions_for_pairs.csv - simple tabular data)
[Future] Pipeline 03: Pair Generation
```

## Benefits

1. **Reduced size** - 10 columns vs 30+
2. **Only relevant data** - Train split, filtered mentions only
3. **Organized structure** - Clear hierarchical grouping
4. **Ready for pairs** - All metadata needed for pair generation
5. **Validated** - Ensures data quality before pair generation
