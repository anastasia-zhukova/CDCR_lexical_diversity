# ms-coref

MS Loss training and evaluation for cross-document coreference resolution (CDCR) over the [AnZhu/uCDCR](https://huggingface.co/datasets/AnZhu/uCDCR) benchmark.

---

## Scripts

### `train.py` — Training

Fine-tunes a RoBERTa-large encoder with **Multi-Similarity Loss** and **P batch sampling** for CDCR.

**Training modes** (set via `TRAIN_MODE` constant at the top of the file):

| Mode | Train on | Evaluate on |
|---|---|---|
| `single` | One dataset (`SINGLE_TRAIN_DATASET`) | That dataset's val split |
| `hypercoref_main` | HyperCoref | All 6 datasets |
| `mix` | All 6 datasets combined | All 6 datasets |

**Usage:**
```bash
python train.py \
  --data-dir data/ \
  --output-dir /path/to/output/ \
  --num-epochs 25 \
  --batch-size 2048 \
  --lr 2e-5
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `data/` | Directory containing `ucdcr_mentions_processed.parquet` |
| `--output-dir` | from `config.yaml` | Where checkpoints and metrics logs are saved |
| `--num-epochs` | `25` | Number of training epochs |
| `--batch-size` | `2048` | Total batch size across all GPUs |
| `--micro-batch-size` | `128` | Forward pass chunk size per GPU (memory control) |
| `--lr` | `2e-5` | Learning rate |
| `--mlp-dim` | `1024` | Hidden dim of the projection MLP |
| `--ms-alpha` | `2.0` | MS Loss α (positive weighting) |
| `--ms-beta` | `50.0` | MS Loss β (negative weighting) |
| `--ms-lambda` | `0.5` | MS Loss margin λ |
| `--val-every` | `1` | Validate every N epochs |
| `--save-every` | `5` | Save checkpoint every N epochs |
| `--run-suffix` | auto-derived | Subdirectory name for this run |
| `--world-size` | `1` | Number of GPUs (auto-detected with `torchrun`) |

**Outputs** (under `--output-dir/<run-suffix>/`):
- `checkpoints/best_model.pt` — best checkpoint by avg pd_LEA F1
- `checkpoints/epoch_<N>.pt` — periodic checkpoints
- `metrics_log.jsonl` — per-epoch metrics for all validation datasets

---

### `conll_eval.py` — Evaluation

Takes a trained checkpoint and evaluates it on the test (or val) split of one or more datasets. Produces CoNLL-formatted files for use with the Perl scorer and a per-subtopic metrics CSV.

**Usage:**
```bash
python conll_eval.py \
  --checkpoint /path/to/best_model.pt \
  --datasets WECEng ECBplus NP4E \
  --split test \
  --data-dir data/ \
  --output-dir conll_output/
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | from `config.yaml` | Path to `.pt` model weights |
| `--datasets` | all 6 datasets | Space-separated list of datasets to evaluate |
| `--split` | `test` | `val` or `test` |
| `--data-dir` | `data/` | Directory containing `ucdcr_mentions_processed.parquet` |
| `--output-dir` | `conll_output/` | Root directory for output files |
| `--setting` | hardcoded `SETTING` | Experiment name used as a subdirectory label |
| `--max-length` | `128` | Token sequence length |
| `--mlp-dim` | `1024` | Must match the value used during training |
| `--batch-size` | `256` | Inference batch size |

**Output directory structure:**
```
{output_dir}/{setting}/{split}/{dataset}/{entity_type}/{subtopic_id}/
    key.conll       — gold clusters
    response.conll  — predicted clusters
metrics.csv         — per-subtopic MUC / B³ / CEAFe / CoNLL / LEA scores
```

Entity types: `entities` (NAM/NOM/PRO), `events`, `mix` (all non-singleton mentions).

**Metrics reported per subtopic:**
`MUC`, `B³`, `CEAFe`, `CoNLL F1` (avg of the three), `LEA`, `pd_LEA`

---

## Configuration — `config.yaml`

Local paths and flags, not committed to version control. Copy from `config.yaml.example`:

```yaml
output_dir: /path/to/training/output
checkpoint: /path/to/best_model.pt

# When true, filters out noisy topics before training and evaluation:
#   HyperCoref: Sport, Sports
#   WECEng:     Awards, Beauty Pageant, Concert, Contest, Festival, Solar Eclipse
# Set to false to disable all topic filtering.
exclude_topics: true
```

Both `train.py` and `conll_eval.py` read `exclude_topics` from this file at startup, so changing it once applies to both scripts.
