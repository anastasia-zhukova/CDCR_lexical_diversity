# Topic Filtering Guide

## Overview

The pipeline now supports topic filtering with a **flagging approach** using an `is_filtered` column in the unified parquet file.

## Design Decision: Single File with Boolean Flag

**Approach**: One parquet file with an `is_filtered` boolean column
- `True` = Mention passes filter (kept for analysis)
- `False` = Mention filtered out (excluded from analysis)

**Why this approach?**
- ✅ Single source of truth
- ✅ Fast filtering with Parquet columnar storage
- ✅ Preserves context (can analyze what was filtered out)
- ✅ Flexible (easy to change filters)
- ✅ Efficient (one boolean column is minimal overhead)

**Alternatives considered:**
- ❌ Two separate files - data duplication
- ❌ Filtered + discarded files - maintenance overhead
- ❌ Parquet views - not natively supported

## Configuration

### Topic Filters (parameters_00_data_exploration.yml)

```yaml
topic_filters:
  HyperCoref:
    - "us"
    - "politics"
    - "international"

  WECEng:
    - "Meetings"
    - "Civilian Attack"
    - "Airliner Accident"
    - "Earthquake"
    - "News Event"
    - "Terrorist Attack"
    - "Wildfire"
    - "Flood"
    - "Weapons Test"
    - "Eruption"
    - "Oilspill"
    - "Rail Accident"
```

**Important:**
- Datasets **not listed** in `topic_filters` → **ALL topics kept** (is_filtered=True for all)
- Datasets **listed** in `topic_filters` → **ONLY specified topics kept**

## Updated Pipeline Flow

```
1. load_all_mentions_node
   ↓ (all_mentions_raw - no filtering)

2. apply_topic_filters_node
   ↓ (all_mentions_df - with is_filtered column)

3. Parallel operations:
   ├─→ create_summary_table_node → summary_table.csv
   ├─→ compute_statistics_node → dataset_statistics.json
   └─→ save_parquet_node → all_mentions.parquet
```

## New Statistics in dataset_statistics.json

### Top Level
```json
{
  "total_mentions": 204068,
  "filtered_mentions": 74611,
  "discarded_mentions": 129457,
  "filtering_applied": true,
  ...
}
```

### By Dataset
```json
"by_dataset": {
  "HyperCoref": {
    "total_mentions": 49911,
    "filtered_mentions": 17629,
    "discarded_mentions": 32282,
    "entity_mentions": 0,
    "event_mentions": 49911,
    "filtered_entity_mentions": 0,
    "filtered_event_mentions": 17629,
    "discarded_entity_mentions": 0,
    "discarded_event_mentions": 32282,
    ...
  }
}
```

### By Entity Style
```json
"by_entity_style": {
  "entity": 102025,
  "event": 102043
},
"by_entity_style_filtered": {
  "kept": {
    "entity": 37305,
    "event": 37306
  },
  "discarded": {
    "entity": 64720,
    "event": 64737
  }
}
```

## Usage

### Run the Pipeline
```bash
kedro run --pipeline 00_data_exploration
```

### Load All Data
```python
import pandas as pd

# Load parquet with filtering info
df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

# View filtering stats
print(df["is_filtered"].value_counts())
# True     74611  (kept)
# False   129457  (discarded)
```

### Work with Filtered Data Only
```python
# Get only mentions that passed the filter
filtered_df = df[df["is_filtered"]]

print(f"Working with {len(filtered_df):,} filtered mentions")
```

### Analyze Discarded Data
```python
# Get mentions that were filtered out
discarded_df = df[~df["is_filtered"]]

# What topics were filtered out in HyperCoref?
hypercoref_discarded = discarded_df[discarded_df["dataset_name"] == "HyperCoref"]
print(hypercoref_discarded["topic"].value_counts())
```

### Filter by Dataset and Status
```python
# HyperCoref mentions that passed filter
hypercoref_kept = df[
    (df["dataset_name"] == "HyperCoref") &
    (df["is_filtered"])
]

# WECEng mentions that were filtered out
weceng_discarded = df[
    (df["dataset_name"] == "WECEng") &
    (~df["is_filtered"])
]
```

### Work with Unfiltered Datasets
```python
# Datasets without filters (all is_filtered=True)
unfiltered_datasets = ["ECBplus", "CD2CR", "CEREC", "FCC", "GVC", "NP4E", "NiDENT"]

# Get all ECBplus data (no filtering applied)
ecbplus = df[df["dataset_name"] == "ECBplus"]
print(f"ECBplus: {(ecbplus['is_filtered']).sum():,} / {len(ecbplus):,} kept (100%)")
```

## Example: Compare Before/After Filtering

```python
import pandas as pd
import json

df = pd.read_parquet("data/02_intermediate/all_mentions.parquet")

with open("data/02_intermediate/dataset_statistics.json") as f:
    stats = json.load(f)

print("Filtering Impact:")
print(f"  Total:     {stats['total_mentions']:,} mentions")
print(f"  Kept:      {stats['filtered_mentions']:,} ({stats['filtered_mentions']/stats['total_mentions']*100:.1f}%)")
print(f"  Discarded: {stats['discarded_mentions']:,} ({stats['discarded_mentions']/stats['total_mentions']*100:.1f}%)")

print("\nBy Dataset:")
for dataset, ds_stats in stats['by_dataset'].items():
    if 'filtered_mentions' in ds_stats:
        total = ds_stats['total_mentions']
        kept = ds_stats['filtered_mentions']
        discarded = ds_stats['discarded_mentions']
        print(f"  {dataset:15} {total:7,} → {kept:7,} ({kept/total*100:5.1f}% kept)")
```

## Benefits of This Approach

### 1. Performance
```python
# Very fast - Parquet reads only the is_filtered column
filtered = df[df["is_filtered"]]  # ~2 seconds for 200K rows
```

### 2. Flexibility
```python
# Easy to try different filters without reloading
df_strict = df[df["is_filtered"] & (df["split"] == "train")]
df_lenient = df  # All data, ignore filtering
```

### 3. Analysis
```python
# Compare filtered vs all data
print("Topic distribution (all data):")
print(df[df["dataset_name"] == "HyperCoref"]["topic"].value_counts())

print("\nTopic distribution (filtered only):")
filtered = df[df["is_filtered"]]
print(filtered[filtered["dataset_name"] == "HyperCoref"]["topic"].value_counts())
```

### 4. Debugging
```python
# Why was this mention filtered out?
mention = df[df["mention_id"] == "some_id"].iloc[0]
if not mention["is_filtered"]:
    print(f"Filtered out: dataset={mention['dataset_name']}, topic={mention['topic']}")
```

## File Structure

```
data/02_intermediate/
├── all_mentions.parquet          # Single file with is_filtered column
├── dataset_statistics.json       # Stats with filtering breakdown
└── summary_table.csv             # Quick reference
```

## Updating Filters

To change the topic filters:

1. Edit `conf/base/parameters_00_data_exploration.yml`
2. Update the `topic_filters` section
3. Re-run the pipeline: `kedro run --pipeline 00_data_exploration`

The parquet file will be regenerated with updated `is_filtered` values.

## Notes

- **All mentions are preserved** - nothing is deleted
- **Filtering is reversible** - change filters and rerun
- **Minimal overhead** - one boolean column adds ~1% to file size
- **Fast queries** - Parquet's columnar format makes `df[df["is_filtered"]]` very efficient
