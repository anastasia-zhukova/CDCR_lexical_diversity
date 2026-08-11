# ms-coref

Training and evaluation for cross-document coreference resolution (CDCR) over the [AnZhu/uCDCR](https://huggingface.co/datasets/AnZhu/uCDCR) benchmark.

Supports two pluggable training regimes, selected via `loss_type` in `config.yaml` (see `core/coref_training_core.py`):

| `loss_type` | Loss | Batch construction |
|---|---|---|
| `ms_loss` (default) | `MultiSimilarityLoss` | Dynamic **PK batch sampler** (`sampler.py`) — chain-labeled batches, PD/difficulty-weighted |
| `contrastive` | `ContrastiveLoss` (margin-based) | Static **SBERT MST + KNN pair sampler** (`pair_sampler.py`), ported from cl-coref-orig |

Both regimes train the exact same `CorefEncoder` architecture and are evaluated with the same per-subtopic pd_LEA/CoNLL protocol — only the loss function and how training batches are built differ.

---

## Code structure

```
ms-coref/
├── config.yaml               # local config — controls almost everything (see below)
├── config.yaml.example       # template to copy from
├── sampler.py                # ms_loss regime: PK batch sampler + PairDataset (contrastive regime's dataset)
├── pair_sampler.py           # contrastive regime: SBERT MST-positive + KNN-hard-negative pair sampler
├── core/                     # shared, import-only modules — no CLI, no __main__
│   ├── coref_training_core.py  #   MultiSimilarityLoss + ContrastiveLoss, label encoder, data
│   │                             #   loading/filtering, model+data construction, LR schedule,
│   │                             #   train-epoch loop (both regimes), and validation evaluation
│   │                             #   shared by both train.py and ray_tune_sweep.py
│   ├── coref_utils.py           #   CorefEncoder model, embedding computation,
│   │                             #   per-subtopic / clustering evaluation helpers
│   └── metrics.py                #   MUC / B³ / CEAFe / CoNLL / LEA / pd_LEA metric functions
├── data_preparation.py       # entry point: build the processed parquet dataset
├── train.py                  # entry point: full (DDP) training run
├── ray_tune_sweep.py         # entry point: Ray Tune HPO sweep
├── eval_checkpoint.py        # entry point: evaluate one checkpoint → CoNLL + metrics CSV
├── eval_all_checkpoints.py   # entry point: run eval_checkpoint.py over every saved run
└── results_summary.py        # entry point: aggregate result CSVs into summary tables
```

**`core/` holds only shared building blocks.** Nothing in `core/` has a CLI or
reads `config.yaml` itself — it takes everything it needs as plain function
arguments. This keeps it a true "auxiliary" layer: any script can import from
it without inheriting another script's configuration or CLI concerns. The
five other top-level scripts are the actual entry points; `train.py` and
`ray_tune_sweep.py` in particular used to duplicate a large amount of setup
logic (loss, encoder, data loading, model/optimizer/scheduler construction,
training-epoch loop, validation) — all of that now lives once in
`core/coref_training_core.py` and `core/coref_utils.py`, and both scripts
just call into it.

---

## Scripts

### `train.py` — Training

Fine-tunes a RoBERTa-large encoder for CDCR using either training regime
(set via `loss_type` in `config.yaml`, default `ms_loss`).

**Training modes** (set via `train_mode` in `config.yaml`):

| Mode | Train on | Evaluate on |
|---|---|---|
| `single` | One dataset (`single_train_dataset`) | That dataset's val split |
| `hypercoref_main` | HyperCoref | All configured `datasets` |
| `mix` | All configured `datasets` combined | All configured `datasets` |

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
| `--ms-alpha` | `2.0` | MS Loss α (positive weighting) — `loss_type: ms_loss` only |
| `--ms-beta` | `50.0` | MS Loss β (negative weighting) — `loss_type: ms_loss` only |
| `--ms-lambda` | `0.5` | MS Loss margin λ — `loss_type: ms_loss` only |
| `--contrastive-margin` | `0.7` | ContrastiveLoss margin (paper: 0.4 events, 0.7 entities) — `loss_type: contrastive` only |
| `--contrastive-sbert-model` | `sentence-transformers/all-mpnet-base-v2` | SBERT model for pair sampling — `loss_type: contrastive` only |
| `--contrastive-k-negatives` | `20` | Hard negatives per anchor mention — `loss_type: contrastive` only |
| `--val-every` | `1` | Validate every N epochs |
| `--save-every` | `5` | Save checkpoint every N epochs |
| `--run-suffix` | auto-derived | Subdirectory name for this run |
| `--world-size` | `1` | Number of GPUs (auto-detected with `torchrun`) |

**Outputs** (under `--output-dir/<run-suffix>/`):
- `checkpoints/best_model.pt` — best checkpoint by avg pd_LEA F1
- `checkpoints/epoch_<N>.pt` — periodic checkpoints
- `metrics_log.jsonl` — per-epoch metrics for all validation datasets

---

### `ray_tune_sweep.py` — Hyperparameter sweep

Runs a Ray Tune HPO sweep (ASHA early stopping) over the search space for the
configured `loss_type`: for `ms_loss`, `lr`, `ms_beta`, `ms_lambda`, `tau`,
`ms_alpha`, `warmup_frac`, `weight_decay`; for `contrastive`, `lr`,
`contrastive_margin`, `warmup_frac`, `weight_decay`. Each trial is a
single-GPU training run reporting `pd_lea_f1` / `conll_f1` per epoch for pruning.

It shares the exact same data loading, model/data construction, LR schedule,
training-epoch loop, and validation logic as `train.py` (both import them from
`core/coref_training_core.py`) — the sweep script itself only wires that shared
logic into a Ray Tune trial and search space, and reads `datasets`, `train_mode`,
`exclude_topics`, etc. from `config.yaml` just like `train.py` does.

**Usage:**
```bash
python ray_tune_sweep.py \
  --data-dir data/ \
  --output-dir /path/to/sweep/output/ \
  --num-trials 50 \
  --num-epochs 25
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | from `config.yaml` (`data_dir`) | Directory containing `ucdcr_mentions_processed.parquet` |
| `--output-dir` | from `config.yaml` (`output_dir`) | Ray Tune's `storage_path` for trial results |
| `--num-trials` | `50` | Number of HPO trials (search-space samples) |
| `--num-epochs` | `25` | Epochs per trial (also `max_t` for ASHA) |

**Output:** Ray Tune's own trial directories under `--output-dir`, plus a
best-trial summary (config + metrics) printed at the end of the run.

---

### `eval_checkpoint.py` — Evaluation

Takes a trained checkpoint and evaluates it on the test (or val) split of one or more datasets. Produces CoNLL-formatted files for use with the Perl scorer and a per-subtopic metrics CSV.

**Usage:**
```bash
python eval_checkpoint.py \
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

### Other entry points

- **`data_preparation.py`** — downloads uCDCR and builds
  `ucdcr_mentions_processed.parquet` (mention/document text context + phrasing
  diversity fields) consumed by `train.py`, `ray_tune_sweep.py`, and
  `eval_checkpoint.py`.
- **`eval_all_checkpoints.py`** — walks `results_root` (from `config.yaml`),
  finds every saved run, and invokes `eval_checkpoint.py` on each one across
  all 12 CDCR datasets (test split).
- **`results_summary.py`** — aggregates the per-run metrics CSVs produced by
  `eval_checkpoint.py` into CoNLL F1 / LEA F1 summary tables (models × datasets).

---

## Configuration — `config.yaml`

**`config.yaml` is the single source of truth for this project's experiment
configuration.** It is local-only (gitignored, not committed) — copy it from
`config.yaml.example` and edit the paths/values for your machine. `train.py`
and `ray_tune_sweep.py` both read the full set of keys below (`datasets`,
`train_mode`, `single_train_dataset`, `loss_type`, `exclude_topics` + its
topic-exclusion sets, `eval_threshold`) at startup and pass the resolved
values into the shared `core/` functions as plain arguments — `core/` itself
never reads this file. `eval_checkpoint.py` / `eval_all_checkpoints.py` read the same file for
their own settings (`checkpoint`, `results_root`, `data_dir`, `exclude_topics`,
`all_datasets`, etc.), though they currently keep their own copies of the
per-dataset topic-exclusion sets and default dataset list rather than sharing
`datasets`/`hypercoref_exclude_topics`/`weceng_exclude_topics` with `train.py`.

```yaml
# Paths
output_dir: /path/to/training/output
checkpoint: /path/to/best_model.pt

# Topic exclusion: when true, filters out noisy topics per dataset before
# training and evaluation. Applies to HyperCoref (Sport/Sports) and WECEng
# (Awards, Beauty Pageant, Concert, Contest, Festival, Solar Eclipse).
# Set to false to run without any topic exclusion.
exclude_topics: true

# Per-dataset topic exclusion sets (only applied when exclude_topics: true).
hypercoref_exclude_topics: [Sport, Sports]
weceng_exclude_topics: [Awards, Beauty Pageant, Concert, Contest, Festival, Solar Eclipse]

# Subset of datasets to use for training/validation (from uCDCR's 12).
datasets: [HyperCoref, NP4E, NewsWCL50, NIdent, ECBplus, ECBplusMETAm]

# Training mode: "single", "hypercoref_main", "mix" (see train.py section above)
train_mode: mix
single_train_dataset: HyperCoref   # only used when train_mode == "single"

# Training regime: "ms_loss" (default) or "contrastive" (see train.py section above)
loss_type: ms_loss

# Fixed clustering threshold for cosine-distance evaluation.
eval_threshold: 0.5

# Full CoNLL evaluation configuration (used by eval_checkpoint.py / eval_all_checkpoints.py)
results_root: /path/to/results_and_weights
data_dir: /path/to/data
repo_dir: /path/to/ms-coref
split: test
batch_size: 256
max_length: 128
mlp_dim: 1024

# All 12 CDCR datasets for full CoNLL evaluation
all_datasets: [HyperCoref, NP4E, NewsWCL50, NIdent, ECBplus, ECBplusMETAm, WECEng, GVC, FCC, MEANTIME, CD2CR, CEREC]
```

Changing `train_mode`, `datasets`, `loss_type`, `exclude_topics`, or
`eval_threshold` once in `config.yaml` applies consistently across `train.py`
and `ray_tune_sweep.py` without editing any code.
