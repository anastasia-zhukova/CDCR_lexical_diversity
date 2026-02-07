# Dataset Architecture Documentation

## Overview

The codebase has been refactored to provide a modular, flexible architecture for loading and combining multiple CDCR datasets with per-dataset restrictions.

## Architecture

### CDCR_Dataset (Single Dataset Loader)

**Purpose**: Loads a single CDCR dataset from a single path with optional restrictions.

**Key Features**:
- Loads ONE dataset from ONE path
- Optional topic filtering (include only specific topics)
- Optional max mentions limit (randomly sample N mentions)
- Uses simple `topic_id` as keys internally (no dataset prefix)

**Constructor**:
```python
CDCR_Dataset(
    name: str,                          # Dataset name (e.g., "ECBplus")
    data_path: str,                     # Single path (e.g., "data/ECBplus/train")
    context_method: str,                # Method for creating mention context
    special_mention_tokens: tuple,      # Mention boundary tokens
    predicted_topics: bool = False,     # Use predicted topics
    topic_filter: List[str] = None,     # List of topic_ids to include
    max_mentions: int = None            # Maximum mentions to load
)
```

**Example**:
```python
# Load ECBplus train split with topics 1-5, max 2000 mentions
dataset = CDCR_Dataset(
    name="ECBplus",
    data_path="data/ECBplus/train",
    context_method="hsu",
    topic_filter=["1", "2", "3", "4", "5"],
    max_mentions=2000
)
```

### CL_Dataset (Multi-Dataset Aggregator)

**Purpose**: Combines multiple CDCR_Dataset instances for contrastive learning.

**Key Features**:
- Creates multiple CDCR_Dataset instances with individual restrictions
- Aggregates mentions from all datasets
- Uses `dataset_topic_key` format (e.g., "ECBplus-1") to prevent namespace collisions
- Maintains backward compatibility with old `data_paths` API
- Provides `cdcr_dataset` property for backward compatibility

**Constructor (New API)**:
```python
CL_Dataset(
    name: str,
    dataset_configs: List[Dict],        # List of dataset configurations
    context_method: str,
    tokenizer,
    special_mention_tokens: tuple,
    mode: str = "train",                # "train" or "inference"
    additional_sampling: bool = False,
    sentence_transformer_model: str = None
)
```

**Dataset Config Format**:
```python
dataset_configs = [
    {
        "name": "ECBplus",
        "path": "data/ECBplus/train",
        "topic_filter": ["1", "2", "3", "4", "5"],  # Optional
        "max_mentions": 2000,                       # Optional
        "predicted_topics": False                   # Optional
    },
    {
        "name": "FCC",
        "path": "data/FCC/train",
        "topic_filter": None,                       # Load all topics
        "max_mentions": 1500,
        "predicted_topics": False
    }
]
```

**Backward Compatibility (Old API)**:
```python
# Old API still works
dataset = CL_Dataset(
    name="ecbplus_train",
    data_paths=["data/ECBplus/train"],  # Old parameter name
    context_method="hsu",
    tokenizer=tokenizer,
    special_mention_tokens=("[E]", "[/E]"),
    mode="train"
)
```

## Configuration System

### data_setups.yaml

Defines which datasets and splits to use, with per-dataset restrictions.

**Format**:
```yaml
0:
  name: "ecbplus"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: null     # null = load all topics
      max_mentions: null     # null = no limit

1:
  name: "ecbplus_fcc_balanced"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: ["1", "2", "3", "4", "5"]
      max_mentions: 2000
    - name: "FCC"
      splits: ["train", "val"]
      topic_filter: null
      max_mentions: 1500
```

**Fields**:
- `name`: Setup identifier
- `datasets`: List of dataset configurations
  - `name`: Dataset name (must match directory in data/)
  - `splits`: List of splits to load (e.g., ["train", "val"])
  - `topic_filter`: List of topic_ids to include, or `null` for all topics
  - `max_mentions`: Maximum mentions per split, or `null` for no limit

### default_config.yaml

Training hyperparameters and settings.

**Relevant Field**:
```yaml
data:
  data_setup: 0  # Reference to data_setups.yaml
  # ... other settings
```

## How It Works

### Single Dataset Scenario

When loading a single dataset:

1. **datamodule.py** reads `data_setup` from config
2. Parses `data_setups.yaml` to get dataset specifications
3. Creates `dataset_configs` list
4. **CL_Dataset** creates one **CDCR_Dataset** instance
5. **CDCR_Dataset** loads mentions, applies topic_filter and max_mentions
6. **CL_Dataset** aggregates (trivial for single dataset)

**Topic Keys**: Simple `topic_id` (e.g., "1", "2", "3")

### Multi-Dataset Scenario

When loading multiple datasets:

1. **datamodule.py** creates `dataset_configs` with restrictions for each dataset
2. **CL_Dataset** creates multiple **CDCR_Dataset** instances
3. Each **CDCR_Dataset** applies its own topic_filter and max_mentions
4. **CL_Dataset** aggregates mentions from all datasets
5. Uses `dataset_topic_key` format (e.g., "ECBplus-1", "FCC-0") to prevent collisions

**Topic Keys**: Namespaced `dataset_topic_key` (e.g., "ECBplus-1", "FCC-0")

### Evaluation

The evaluation system automatically detects single vs. multi-dataset scenarios:

**Single Dataset**:
- Uses simple `topic_id` for grouping

**Multi-Dataset**:
- Uses `dataset_topic_key` to prevent topic ID collisions
- Extracts dataset name from mention's `source_path`

## Usage Examples

### Example 1: Train on ECBplus only

```yaml
# In data_setups.yaml
0:
  name: "ecbplus"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: null
      max_mentions: null
```

```bash
python train.py --config configs/default_config.yaml --data_setup 0
```

### Example 2: Train on ECBplus + FCC with balanced counts

```yaml
# In data_setups.yaml
1:
  name: "ecbplus_fcc_balanced"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: null
      max_mentions: 2000
    - name: "FCC"
      splits: ["train", "val"]
      topic_filter: null
      max_mentions: 2000
```

```bash
python train.py --config configs/default_config.yaml --data_setup 1
```

### Example 3: Train on specific topics only

```yaml
# In data_setups.yaml
3:
  name: "ecbplus_topics_1_5"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: ["1", "2", "3", "4", "5"]
      max_mentions: null
```

```bash
python train.py --config configs/default_config.yaml --data_setup 3
```

### Example 4: Multiple datasets with different restrictions

```yaml
# In data_setups.yaml
5:
  name: "multi_dataset_custom"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
      max_mentions: 1500
    - name: "FCC"
      splits: ["train", "val"]
      topic_filter: null
      max_mentions: 1000
    - name: "NP4E"
      splits: ["train", "val"]
      topic_filter: ["1", "2", "3", "4", "5"]
      max_mentions: 800
```

```bash
python train.py --config configs/default_config.yaml --data_setup 5
```

## Benefits

1. **Modularity**: Clear separation between single-dataset loading and multi-dataset aggregation
2. **Flexibility**: Per-dataset topic filtering and mention limits
3. **Config-driven**: All restrictions specified in YAML, not hardcoded
4. **Namespace safety**: `dataset_topic_key` prevents topic ID collisions across datasets
5. **Backward compatible**: Existing code continues to work
6. **No friction**: Easy to create new dataset combinations without code changes

## Migration Guide

### From Old to New API

**Old Code**:
```python
dataset = CL_Dataset(
    name="ecbplus_train",
    data_paths=["data/ECBplus/train"],
    context_method="hsu",
    tokenizer=tokenizer,
    special_mention_tokens=("[E]", "[/E]"),
    mode="train"
)
```

**New Code** (equivalent):
```python
dataset = CL_Dataset(
    name="ecbplus_train",
    dataset_configs=[{
        "name": "ECBplus",
        "path": "data/ECBplus/train"
    }],
    context_method="hsu",
    tokenizer=tokenizer,
    special_mention_tokens=("[E]", "[/E]"),
    mode="train"
)
```

**Note**: The old API still works due to backward compatibility, but new code should use `dataset_configs`.

## Implementation Details

### Topic Filtering

Applied in `CDCR_Dataset.load_mentions()`:
```python
if self.topic_filter and new_mention.topic_id not in self.topic_filter:
    continue  # Skip this mention
```

### Max Mentions Limiting

Applied in `CDCR_Dataset._apply_max_mentions_limit()`:
- Randomly samples mentions (seed=42 for reproducibility)
- Updates all derived structures (mentions_by_topic, mentions_by_sentence, etc.)

### Dataset Aggregation

Performed in `CL_Dataset._aggregate_datasets()`:
- Merges mentions from all CDCR_Datasets
- Rebuilds topic structures with `dataset_topic_key` namespace
- Creates unified `gold_clusters_df` with dataset_name column

### Evaluation Detection

Performed in `lightning_module.py`:
```python
use_dataset_prefix = len(getattr(val_dataset, 'cdcr_datasets', [cdcr_dataset])) > 1
```

## Testing Checklist

- [x] Single dataset load (ECBplus only)
- [x] Multi-dataset load (ECBplus + FCC)
- [x] Topic filtering (ECBplus topics 1-5)
- [x] Max mentions limiting
- [x] Backward compatibility with old data_paths API
- [ ] Training full pipeline (requires actual training run)
- [ ] Evaluation with multi-dataset (requires actual training run)

## Troubleshooting

### Issue: "Mention does not have a matching doc key"

**Cause**: topic_filter excluded all documents for a topic that has mentions.

**Solution**: Ensure topic_filter includes topics that have both documents and mentions.

### Issue: Unexpected mention counts

**Cause**: max_mentions is applied per CDCR_Dataset, not globally.

**Solution**: If combining multiple datasets, set max_mentions for each dataset individually.

### Issue: Topic ID collisions in evaluation

**Cause**: Multiple datasets with overlapping topic IDs (e.g., ECBplus-1 and FCC-1).

**Solution**: The system automatically uses dataset_topic_key in multi-dataset scenarios. If seeing issues, check that `use_dataset_prefix=True` is being used.
