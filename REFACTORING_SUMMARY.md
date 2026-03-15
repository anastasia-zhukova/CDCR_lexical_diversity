# Dataset Architecture Refactoring Summary

## Problem Statement

The original architecture had a fundamental design issue where `CDCR_Dataset` was designed to load multiple datasets from multiple paths, but topic filtering was global and used a format like `"DatasetName-TopicID"`. This created problems:

1. **No per-dataset control**: Couldn't apply different topic filters to different datasets
2. **Manual topic key construction**: Required knowing dataset names and manually specifying `"ECBplus-1"`, `"FCC-0"` format
3. **No mention limiting**: Couldn't restrict the number of mentions per dataset
4. **Coupling**: Single class doing both loading and aggregation
5. **Configuration friction**: Had to modify code to change dataset combinations

## Solution

Refactored the architecture to separate concerns:

- **CDCR_Dataset**: Single dataset loader (ONE dataset from ONE path)
- **CL_Dataset**: Multi-dataset aggregator (combines multiple CDCR_Datasets)

## Changes Made

### 1. CDCR_Dataset (dataset.py:107-260)

**Changed to single-path loader with restrictions**:
- `data_path`: Now accepts only a single string (not list)
- `topic_filter`: Optional list of topic_ids to include (e.g., `["1", "2", "3"]`)
- `max_mentions`: Optional maximum number of mentions to load
- Internal keys use simple `topic_id` (no dataset prefix)
- Added `_apply_max_mentions_limit()` method for random sampling

**Example**:
```python
dataset = CDCR_Dataset(
    name="ECBplus",
    data_path="data/ECBplus/train",
    context_method="hsu",
    topic_filter=["1", "2", "3", "4", "5"],
    max_mentions=2000
)
```

### 2. CL_Dataset (dataset.py:318-570)

**Changed to multi-dataset aggregator**:
- New parameter: `dataset_configs` (list of dataset specifications)
- Backward compatible: Still accepts `data_paths` (auto-converts to new format)
- Creates multiple CDCR_Dataset instances internally
- Aggregates mentions with `dataset_topic_key` namespace (e.g., "ECBplus-1")
- Added `_aggregate_datasets()` method
- Added `cdcr_dataset` property for backward compatibility

**Example**:
```python
dataset = CL_Dataset(
    name="multi_dataset",
    dataset_configs=[
        {
            "name": "ECBplus",
            "path": "data/ECBplus/train",
            "topic_filter": ["1", "2", "3", "4", "5"],
            "max_mentions": 2000
        },
        {
            "name": "FCC",
            "path": "data/FCC/train",
            "topic_filter": None,
            "max_mentions": 1500
        }
    ],
    context_method="hsu",
    tokenizer=tokenizer,
    special_mention_tokens=("[E]", "[/E]"),
    mode="train"
)
```

### 3. Configuration System

**Updated data_setups.yaml** (configs/data_setups.yaml):
- Added `topic_filter` field (list of topic_ids or null)
- Added `max_mentions` field (integer or null)
- Backward compatible: Existing configs work without these fields

**Example**:
```yaml
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

**Updated datamodule.py** (datamodule.py:49-81):
- Parses new `topic_filter` and `max_mentions` fields
- Constructs `dataset_configs` list for CL_Dataset
- Passes per-dataset restrictions correctly

### 4. Evaluation System

**Updated evaluation.py** (evaluation.py:33-62):
- Modified `return_embeddings_by_topic()` to accept `use_dataset_prefix` parameter
- When True, extracts dataset name from mention's `source_path`
- Creates `dataset_topic_key` format for multi-dataset scenarios

**Updated lightning_module.py** (lightning_module.py:69-79):
- Detects multi-dataset scenarios automatically
- Passes `use_dataset_prefix=True` when multiple datasets are loaded
- Ensures correct topic grouping during evaluation

## Key Benefits

1. **Modularity**: Clear separation of concerns (loading vs. aggregation)
2. **Flexibility**: Per-dataset topic filtering and mention limits
3. **Config-driven**: All restrictions in YAML, no code changes needed
4. **Namespace safety**: Automatic dataset_topic_key prevents collisions
5. **Backward compatible**: Existing code works without modifications
6. **No friction**: Easy to create new dataset combinations

## Usage

### Quick Start

1. **Edit data_setups.yaml**:
```yaml
3:
  name: "my_custom_setup"
  datasets:
    - name: "ECBplus"
      splits: ["train", "val"]
      topic_filter: ["1", "2", "3"]  # Only these topics
      max_mentions: 1000              # Limit mentions
    - name: "FCC"
      splits: ["train", "val"]
      topic_filter: null              # All topics
      max_mentions: 800
```

2. **Run training**:
```bash
python train.py --config configs/default_config.yaml --data_setup 3
```

That's it! No code changes required.

## Files Modified

1. **dataset.py**: Refactored CDCR_Dataset and CL_Dataset
2. **datamodule.py**: Updated to parse new config format
3. **evaluation.py**: Added multi-dataset support
4. **lightning_module.py**: Auto-detects multi-dataset scenarios
5. **configs/data_setups.yaml**: Added topic_filter and max_mentions fields

## Files Created

1. **configs/data_setups_examples.yaml**: Example configurations
2. **DATASET_ARCHITECTURE.md**: Comprehensive documentation
3. **REFACTORING_SUMMARY.md**: This file

## Backward Compatibility

All existing code continues to work:

```python
# Old API (still works)
dataset = CL_Dataset(
    name="ecbplus_train",
    data_paths=["data/ECBplus/train"],
    context_method="hsu",
    tokenizer=tokenizer,
    special_mention_tokens=("[E]", "[/E]"),
    mode="train"
)

# Internally converts to:
dataset_configs = [{
    "name": "ECBplus",
    "path": "data/ECBplus/train"
}]
```

## Next Steps

1. **Test the implementation**: Run training with different data_setup configurations
2. **Verify evaluation**: Check that multi-dataset evaluation works correctly
3. **Update predict.py**: If needed, update prediction scripts to use new API
4. **Create more setups**: Add useful dataset combinations to data_setups.yaml

## Example Configurations

See `configs/data_setups_examples.yaml` for ready-to-use examples:
- Setup 3: ECBplus with topics 1-5 only
- Setup 4: ECBplus + FCC with balanced mention counts
- Setup 5: Multi-dataset with different restrictions per dataset
- Setup 6: Small subset for quick testing

## Questions?

See `DATASET_ARCHITECTURE.md` for detailed documentation, or check the code comments in:
- `dataset.py` (lines 107-260 for CDCR_Dataset, 318-570 for CL_Dataset)
- `evaluation.py` (lines 33-62 for multi-dataset support)
