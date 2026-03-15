# Cross-Document Coreference Resolution with Multi-Similarity Loss

A metric learning approach to cross-document coreference resolution (CDCR) using **Multi-Similarity Loss** with **phrasing diversity (PD)-weighted evaluation**. The model learns to embed mentions into a shared space where coreferent mentions are close and non-coreferent mentions are far apart, across documents and datasets.

## Architecture

**RoBERTa-large** encoder with dual MLP projection heads producing 2048-dim L2-normalized embeddings per mention:

- **CLS projection** (1024-dim): `[CLS]` token → 3-layer MLP → document-level context
- **Mention span projection** (1024-dim): sum of `[E]...[/E]` tokens → 3-layer MLP → entity-specific signal

At inference, **agglomerative clustering** (average linkage, cosine distance) groups embeddings using a fixed threshold.

## Training

Training uses **PK batch sampling** with structured negative difficulty:

- 60% chains from the same subtopic (hard negatives)
- 20% from the same topic, different subtopic (medium negatives)
- 20% random chains (easy negatives)
- Remaining slots filled with singletons

Chain sampling is weighted by phrasing diversity: `weight = PD^tau * log(chain_size) / log(dataset_chain_count)`.

### Datasets

Training uses 6 datasets from the [unified CDCR corpus (uCDCR)](https://huggingface.co/datasets/AnZhu/uCDCR):

| Dataset | Description |
|---------|-------------|
| HyperCoref | Hypernym-based entity coreference |
| NP4E | Noun phrase coreference for events |
| NewsWCL50 | High phrasing diversity news coreference |
| NIdent | Named entity identification |
| ECBplus | Event and entity coreference benchmark |
| ECBplusMETAm | ECB+ with metonymic annotations |

### Training modes

Configure via constants at the top of `ms-coref/train.py`:

| Mode | `TRAIN_MODE` | Description |
|------|-------------|-------------|
| Single-dataset | `"single"` | Train on one dataset, evaluate on all 6. Set `SINGLE_TRAIN_DATASET` to choose which. |
| HyperCoref-main | `"hypercoref_main"` | Train on HyperCoref only, evaluate on all 6. |
| Dataset mix | `"mix"` | Train on all 6 datasets combined, evaluate on all 6. |

### Running training

```bash
# Data preparation (downloads uCDCR, builds mention representations)
cd ms-coref
python data_preparation.py

# Training (defaults: tau=1.0, threshold=0.5, batch_size=2048)
python train.py
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--tau` | 1.0 | PD exponent for chain weighting |
| `--batch-size` | 2048 | Total batch size |
| `--micro-batch-size` | 128 | Gradient accumulation micro-batch |
| `--num-epochs` | 50 | Training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--ms-alpha` | 2.0 | MS Loss positive steepness |
| `--ms-beta` | 50.0 | MS Loss negative steepness |
| `--ms-lambda` | 0.5 | MS Loss similarity margin |

### Evaluation

Evaluation uses **pd_LEA F1** (LEA weighted by phrasing diversity) as the primary metric, computed at a fixed cosine distance threshold of 0.5. Standard LEA metrics (precision, recall, F1) are also reported per dataset.

### Inference on NewsWCL50

```bash
python infer_news.py --checkpoint path/to/best_model.pt
```

Runs threshold sweep + error analysis (per-topic, per-mention-type, per-NER-type, chain-level) on NewsWCL50 val and test splits.

## Project structure

```
ms-coref/              # MS Loss metric learning approach (main)
  train.py             # Training loop with PK sampling
  coref_utils.py       # CorefEncoder model, embedding computation, LEA evaluation
  pk_sampler.py        # PK batch sampler, chain index, custom tokenizer
  data_preparation.py  # uCDCR data download and processing
  infer_news.py        # NewsWCL50 inference and error analysis
  threshold_sweep.py   # Post-hoc threshold sweep utility

MTL/                   # Multi-task learning approach (cross-encoder baseline)
  cross_encoder_adapted/  # Pairwise cross-encoder training and evaluation

CL_COREF/              # Coreference evaluation tools (coval scorer)
cl-coref/              # Data exploration and mention preparation pipelines
```

## Previous approach

The **multi-task learning (MTL) cross-encoder** approach is documented in [MTL/README.md](MTL/README.md). It combines CDCR with paraphrase identification using pairwise scoring.
