# Quick Start - Pipeline 01_mentions_preparation

## ⚡ TL;DR

```bash
cd /home/ubuntu/github/CDCR_lexical_diversity/cl-coref
kedro run --pipeline 01_mentions_preparation
```

**Time**: ~27 minutes
**Output**: 116,793 mention representations in `data/03_primary/mention_representations.parquet`

---

## 📦 What It Does

Transforms filtered train mentions into string representations ready for tokenization:

**Input**: 116,793 filtered train mentions (from pipeline 00)
**Output**: String representations with format:
```
"{8 sentences context} [SEP] {sentence with [E] mention [/E] markers}"
```

---

## 📁 Output Files

```
data/03_primary/
├── mention_representations.parquet  (48 MB)  ← Main output
├── mentions_for_pairs.csv          (12 MB)  ← Reference data
├── mention_groups_summary.csv      (2.5 KB) ← Group overview
└── grouping_statistics.json        (34 KB)  ← Statistics
```

---

## ✅ Verify Success

```bash
# Check output file exists and has correct size
ls -lh data/03_primary/mention_representations.parquet

# Verify all mentions processed
python -c "import pandas as pd; \
  df = pd.read_parquet('data/03_primary/mention_representations.parquet'); \
  print(f'✓ {len(df):,} representations created'); \
  print(f'✓ {df.mention_context.str.contains(\"[E]\").sum():,} have [E]'); \
  print(f'✓ {df.mention_context.str.contains(\"[/E]\").sum():,} have [/E]')"
```

Expected output:
```
✓ 116,793 representations created
✓ 116,793 have [E]
✓ 116,793 have [/E]
```

---

## 🔄 Re-run Specific Node

```bash
# Just regenerate string representations
kedro run --pipeline 01_mentions_preparation \
  --from-nodes create_mention_string_representation_node
```

---

## 📊 View Statistics

```bash
# See group breakdown
cat data/03_primary/mention_groups_summary.csv

# See detailed statistics
python -c "import json; \
  print(json.dumps(json.load(open('data/03_primary/grouping_statistics.json')), indent=2))"
```

---

## 🐛 Common Issues

### Issue: OOM Error
**Symptom**: Process killed, exit code 137
**Fix**: Already handled via batch processing. If it persists, reduce batch size in `nodes.py`:
```python
batch_size = 500_000  # Default is 1_000_000
```

### Issue: Parsing Errors
**Symptom**: "invalid syntax" warnings for tokens_number
**Fix**: Already fixed. Ensure you have the latest `nodes.py` code.

### Issue: Missing Output File
**Symptom**: File not created after pipeline run
**Check**:
```bash
# View logs
tail -100 logs/kedro.log

# Check if previous pipeline ran
kedro run --pipeline 00_data_exploration
```

---

## 📖 Documentation

- **Full Details**: See `PIPELINE_02_PAIRS_CREATION_SUMMARY.md`
- **Format Guide**: See `MENTION_REPRESENTATION_GUIDE.md`
- **Configuration**: See `conf/base/parameters_01_mentions_preparation.yml`

---

## 🎯 Next Pipeline

After this completes, you're ready for:
- **Pipeline 03**: Tokenization (convert strings → token IDs)
- **Pipeline 04**: Pair generation (create training pairs)
- **Pipeline 05**: Model training

---

**Status**: ✅ Production-Ready
**Last Run**: February 8, 2026
**Success Rate**: 100% (116,793/116,793)
