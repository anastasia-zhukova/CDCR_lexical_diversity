# Cross-Document Coreference Resolution and Lexical Diversity

Cross-document coreference resolution (CDCR) groups mentions that refer to the
same entity or event across documents. This repository studies how well CDCR
models handle chains with high **phrasing diversity (PD)**, where the same
entity is described in very different words. Models are trained and evaluated on the unified
[uCDCR](https://huggingface.co/datasets/AnZhu/uCDCR) benchmark, and **pd_LEA**
(LEA weighted by phrasing diversity) is reported alongside the standard
MUC / B³ / CEAFe / CoNLL / LEA metrics.

> **Active code: [`ms-coref/`](ms-coref/).** Everything else at the top level
> is earlier work kept for reference (see [Legacy folders](#legacy-folders)).
> Full documentation for `ms-coref` (scripts, CLI arguments, config keys,
> output layout) is in [ms-coref/README.md](ms-coref/README.md).

---

## `ms-coref` at a glance

### Model

A **RoBERTa-large** encoder with two MLP projection heads. Each mention is
embedded as a 2048-dim, L2-normalized vector made by concatenating:

- **CLS projection** (1024-dim): the `<s>` token passed through a 3-layer MLP.
- **Mention-span projection** (1024-dim): the sum of the tokens between the
  `[E]` … `[/E]` markers, passed through a separate 3-layer MLP.

At inference, mentions are grouped by **agglomerative clustering** (average
linkage, cosine distance, fixed threshold from `eval_threshold`).

### Two training regimes: ms-coref and cl-coref

The same encoder can be trained with either approach. You choose with
`loss_type` in `ms-coref/config.yaml`:

| `loss_type` | Approach | Loss | How training batches are built |
|---|---|---|---|
| `ms_loss` (default) | **ms-coref** | Multi-Similarity Loss | Dynamic **PK batch sampler** (`sampler.py`): ~60% of chains from one subtopic (hard negatives), ~20% from the same topic, ~20% random, then singletons. Chains are weighted by `PD^tau · log(chain_size)` and by a per-chain difficulty average updated from the loss. |
| `contrastive` | **cl-coref** (ported from the earlier contrastive-learning model) | Margin-based contrastive loss | Static **pair sampler** (`pair_sampler.py`): positives from a minimum spanning tree over each chain (least similar sentences first, by SBERT); negatives are the K most similar non-coreferent mentions in the same subtopic. |

The encoder, data pipeline, and evaluation are shared, so only the loss
and the way batches are built differ between the two regimes. This makes
them directly comparable.

### Training modes

Set with `train_mode` in `config.yaml`:

| Mode | Trains on | Validates on |
|---|---|---|
| `mix` (default) | All configured `datasets` combined | All configured `datasets` |
| `hypercoref_main` | HyperCoref only | All configured `datasets` |
| `single` | `single_train_dataset` only | That dataset's val split |

The default `datasets` are HyperCoref, NP4E, NewsWCL50, NIdent, ECBplus and
ECBplusMETAm. NewsWCL50 and ECBplusMETAm have **no train split** in uCDCR,
so in `mix` mode they are used for validation only. Evaluation can also run
on all 12 uCDCR datasets (adding WECEng, GVC, FCC, MEANTIME, CD2CR and
CEREC). Noisy topics in HyperCoref and WECEng can be excluded with
`exclude_topics`.

### Evaluation

- **During training**, every validation epoch computes pd_LEA, LEA and CoNLL
  F1 per dataset. The best checkpoint is chosen by `selection_metric`
  (default `pd_lea_f1`), averaged across validation datasets.
- **`eval_checkpoint.py`** evaluates a checkpoint on the test (or val)
  split, clustering at both **subtopic** and **topic** level. It writes
  CoNLL key/response files for the reference Perl scorer and a per-group
  metrics CSV, split into entities, events, and mixed.

### Quick start

```bash
cd ms-coref
pip install -r requirements.txt
cp config.yaml.example config.yaml          # then edit paths / settings

python data_preparation.py                  # download uCDCR → data/ucdcr_mentions_processed.parquet
python train.py --output-dir /path/to/runs  # single GPU; use torchrun for multi-GPU
python eval_checkpoint.py --checkpoint /path/to/runs/<run>/checkpoints/best_model.pt
```

Other entry points: `ray_tune_sweep.py` (hyperparameter sweep with Ray Tune +
ASHA), `eval_all_checkpoints.py` (evaluate every saved run), and
`results_summary.py` (combine results into CoNLL / LEA summary tables). See
[ms-coref/README.md](ms-coref/README.md) for details.

### Layout

```
ms-coref/
├── config.yaml.example     # copy to config.yaml (gitignored); controls datasets, modes, loss_type, …
├── data_preparation.py     # build the processed mention parquet from uCDCR
├── train.py                # training (single GPU or DDP)
├── ray_tune_sweep.py       # hyperparameter sweep
├── eval_checkpoint.py      # test/val evaluation → CoNLL files + metrics CSV
├── eval_all_checkpoints.py # evaluate every saved run
├── results_summary.py      # aggregate results into summary tables
├── sampler.py              # ms_loss: PK batch sampler, datasets, tokenizer
├── pair_sampler.py         # contrastive: MST / KNN pair sampler
├── results_analysis.md     # notes on experimental runs
└── core/                   # shared modules (no CLI): losses, training loop, model, metrics
```

---

## Legacy folders

These folders hold earlier approaches and tooling. They are **no longer
maintained** and may not run against the current data or dependencies. Use
them for reference only.

| Folder | What it was |
|---|---|
| [`MTL/`](MTL/) | Multi-task cross-encoder combining CDCR with paraphrase identification (pairwise scoring). |
| [`PI/`](PI/) | Paraphrase-identification data used by the MTL approach. |
| [`CL_COREF/`](CL_COREF/) | Original siamese contrastive-learning CDCR model, plus the `coval` scorer. The `contrastive` regime in `ms-coref` is a port of this approach, and `ms-coref`'s PD calculation mirrors its scorer. |
| [`cl-coref/`](cl-coref/) | Kedro pipelines for data exploration, mention preparation and pair selection that fed `CL_COREF`. |
| [`DAPT/`](DAPT/) | Domain-adaptive pretraining experiments (cluster job scripts). |

## License

See [LICENSE](LICENSE).
