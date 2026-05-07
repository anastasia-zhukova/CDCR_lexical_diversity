"""
MS Loss training for cross-document coreference resolution.

Uses PK batch sampling with structured negative difficulty,
cross-GPU embedding gathering for large effective batches,
and pd_LEA validation across 6 CDCR datasets.

Training modes (configured via constants at top of file):
  - single:          Train on one dataset, evaluate on all
  - hypercoref_main: Train on HyperCoref, evaluate on all
  - mix:             Train on all datasets combined
"""

import argparse
import contextlib
import json
import logging
import math
import os
import subprocess
import sys
import yaml
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from coref_utils import CorefEncoder, compute_embeddings, evaluate_per_subtopic
from sampler import MentionDataset, build_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Databricks/Spark infrastructure noise
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Subset of datasets to use (from uCDCR's 12)
DATASETS = ["HyperCoref", "NP4E", "NewsWCL50", "NIdent", "ECBplus", "ECBplusMETAm"]

# Read exclude_topics flag from config.yaml (defaults to True if not set).
def _read_exclude_topics_flag():
    base = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as _f:
            return (yaml.safe_load(_f) or {}).get("exclude_topics", True)
    return True

EXCLUDE_TOPICS = _read_exclude_topics_flag()

# Per-dataset topic exclusion sets (only used when EXCLUDE_TOPICS=True).
HYPERCOREF_EXCLUDE_TOPICS = {"Sport", "Sports"}
WECENG_EXCLUDE_TOPICS = {
    "Awards", "Beauty Pageant", "Concert", "Contest", "Festival", "Solar Eclipse"
}

# Training mode: "single", "hypercoref_main", "mix"
#   single          – train on one dataset at a time (loop over DATASETS externally)
#   hypercoref_main – train on HyperCoref, evaluate on all DATASETS
#   mix             – train on all DATASETS combined
TRAIN_MODE = "single"

# When TRAIN_MODE == "single", which dataset to train on.
# Ignored for other modes.  Change this (or loop externally) per run.
SINGLE_TRAIN_DATASET = "ECBplus"

# Fixed clustering threshold for cosine-distance evaluation
EVAL_THRESHOLD = 0.5

def safe_torch_save(obj, path):
    """
    Save via local /tmp then copy to destination.

    Avoids two Databricks issues:
    - inline_container.cc errors from PyTorch zip writes to network filesystems
    - sendfile 2GB limit when copying large checkpoints
    """
    path = Path(path)
    tmp_path = Path("/tmp") / path.name
    torch.save(obj, tmp_path)
    if path.exists():
        path.unlink()
    with open(tmp_path, "rb") as fsrc, open(path, "wb") as fdst:
        while buf := fsrc.read(64 * 1024 * 1024):
            fdst.write(buf)
    tmp_path.unlink()



# ---------------------------------------------------------------------------
# Multi-Similarity Loss
# ---------------------------------------------------------------------------

class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss from Wang et al. (CVPR 2019).

    L = (1/m) * sum_i [
        (1/alpha) * log(1 + sum_{k in P_i} exp(-alpha * (S_ik - lambda)))
      + (1/beta)  * log(1 + sum_{k in N_i} exp( beta * (S_ik - lambda)))
    ]

    where S_ik = cosine_similarity(z_i, z_k), P_i = positive pairs,
    N_i = hard negative pairs mined online.
    """

    def __init__(self, alpha=2.0, beta=50.0, lam=0.5, eps=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.eps = eps  # mining margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) integer class labels.

        Returns:
            Scalar loss, per-anchor loss dict keyed by label (for difficulty tracking).
        """
        # Cosine similarity matrix
        sim_mat = embeddings @ embeddings.t()  # (B, B)

        # Build positive/negative masks
        label_mat = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        pos_mask = label_mat & ~eye_mask
        neg_mask = ~label_mat

        # Use embeddings.sum() * 0 so the tensor is always in the computation
        # graph — avoids "does not require grad" errors on empty batches.
        loss = embeddings.sum() * 0.0
        n_valid = 0
        n_mined_pos = 0
        n_mined_neg = 0
        per_label_loss = defaultdict(list)

        for i in range(len(labels)):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_sim = sim_mat[i, pos_indices]
            neg_sim = sim_mat[i, neg_indices]

            # Online mining: keep negatives harder than easiest positive - eps
            # and positives harder than hardest negative + eps
            neg_mining_mask = neg_sim > pos_sim.min() - self.eps
            pos_mining_mask = pos_sim < neg_sim.max() + self.eps

            if neg_mining_mask.sum() == 0 or pos_mining_mask.sum() == 0:
                continue

            mined_pos_sim = pos_sim[pos_mining_mask]
            mined_neg_sim = neg_sim[neg_mining_mask]

            n_mined_pos += pos_mining_mask.sum().item()
            n_mined_neg += neg_mining_mask.sum().item()

            # Positive term
            pos_term = (1.0 / self.alpha) * torch.log(
                1.0 + torch.sum(torch.exp(-self.alpha * (mined_pos_sim - self.lam)))
            )
            # Negative term
            neg_term = (1.0 / self.beta) * torch.log(
                1.0 + torch.sum(torch.exp(self.beta * (mined_neg_sim - self.lam)))
            )

            anchor_loss = pos_term + neg_term
            loss += anchor_loss
            n_valid += 1

            label_val = labels[i].item()
            per_label_loss[label_val].append(anchor_loss.item())

        if n_valid > 0:
            loss = loss / n_valid

        pair_stats = {"n_mined_pos": n_mined_pos, "n_mined_neg": n_mined_neg, "n_valid_anchors": n_valid}
        return loss, per_label_loss, pair_stats


# CorefEncoder, compute_embeddings, evaluate_clustering live in coref_utils.py


# ---------------------------------------------------------------------------
# Cross-GPU embedding gathering
# ---------------------------------------------------------------------------

class GatherWithGrad(torch.autograd.Function):
    """All-gather that allows gradients to flow back through local embeddings."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ctx.rank = dist.get_rank()
        ctx.world_size = world_size
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Only pass gradient for local shard
        chunk_size = grad_output.size(0) // ctx.world_size
        return grad_output[ctx.rank * chunk_size : (ctx.rank + 1) * chunk_size]


def gather_embeddings(embeddings, labels_int):
    """
    Gather embeddings and labels across all GPUs.

    In single-GPU mode, returns inputs unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return embeddings, labels_int

    all_embeddings = GatherWithGrad.apply(embeddings)

    # Labels don't need gradients — use standard all_gather
    gathered_labels = [torch.zeros_like(labels_int) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_labels, labels_int)
    all_labels = torch.cat(gathered_labels, dim=0)

    return all_embeddings, all_labels


# ---------------------------------------------------------------------------
# Label encoding (string chain IDs -> integer labels for MS Loss)
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Maps string coref_chain IDs to contiguous integer labels."""

    def __init__(self):
        self._map = {}
        self._next = 0

    def encode(self, labels):
        """
        Args:
            labels: list of string chain IDs.

        Returns:
            torch.LongTensor of integer labels.
        """
        int_labels = []
        for label in labels:
            if label not in self._map:
                self._map[label] = self._next
                self._next += 1
            int_labels.append(self._map[label])
        return torch.tensor(int_labels, dtype=torch.long)




def build_val_resources(mention_df, tokenizer, max_length, val_batch_size, num_workers,
                        datasets=None):
    """
    Build per-dataset validation dataloaders, PD tuples, and subtopic IDs.

    Args:
        datasets: optional list of dataset names to include. If None, all datasets are used.

    Returns:
        val_resources: dict of dataset_name -> {
            "dataloader": DataLoader,
            "pd_tuples": list of (head, wordset, wo_stopwords) per mention,
            "subtopic_ids": list of subtopic identifiers per mention,
            "n_mentions": int,
            "n_subtopics": int,
        }
    """
    val_resources = {}
    val_splits = mention_df[
        mention_df["split"].isin(["val", "validation"])
    ]
    if datasets is not None:
        val_splits = val_splits[val_splits["dataset"].isin(datasets)]

    for dataset_name, ds_df in val_splits.groupby("dataset"):
        ds_df = ds_df.reset_index(drop=True)
        dataset = MentionDataset(ds_df, tokenizer, max_length=max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Pre-extract PD tuples aligned with the reset index
        pd_tuples = []
        for _, row in ds_df.iterrows():
            head = row.get("mention_head")
            wordset = row.get("mention_wordset")
            wo_stop = row.get("mention_wo_stopwords")
            if isinstance(wordset, str) and isinstance(wo_stop, str):
                pd_tuples.append((head, wordset, wo_stop))
            else:
                pd_tuples.append((head, None, None))

        subtopic_ids = ds_df["subtopic_id"].tolist()

        val_resources[dataset_name] = {
            "dataloader": dataloader,
            "pd_tuples": pd_tuples,
            "subtopic_ids": subtopic_ids,
            "n_mentions": len(ds_df),
            "n_subtopics": ds_df["subtopic_id"].nunique(),
        }

    return val_resources


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _safe_torch_save(obj, path):
    """Save checkpoint, handling Databricks filesystem limitations.

    Saves to /tmp first (local disk), then copies to the target path
    in chunks. Falls back to keeping in /tmp if the copy fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to /tmp first (always works)
    tmp_dir = Path("/tmp/ms_coref_checkpoints")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / path.name
    torch.save(obj, str(tmp_path))

    # Try copying to target path in chunks
    try:
        chunk_size = 16 * 1024 * 1024  # 16 MB
        with open(str(tmp_path), "rb") as fsrc, open(str(path), "wb") as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
        logger.info(f"  Checkpoint saved to {path}")
    except OSError as e:
        logger.warning(f"  Could not copy to {path}: {e}")
        logger.info(f"  Checkpoint kept at {tmp_path}")


def train(args):
    # --- Silence noisy Databricks / Py4J loggers ---
    for noisy in ["py4j", "py4j.java_gateway", "py4j.clientserver"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_distributed = args.world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0

    is_main = local_rank == 0

    if is_main:
        logger.info(f"Device: {device}, World size: {args.world_size}")

    # --- Load data ---
    data_path = Path(args.data_dir) / "ucdcr_mentions_processed.parquet"
    mention_df = pd.read_parquet(data_path)
    if is_main:
        logger.info(f"Loaded {len(mention_df):,} mentions from {data_path}")

    # Filter to experiment datasets
    mention_df = mention_df[mention_df["dataset"].isin(DATASETS)].copy()
    if is_main:
        logger.info(f"Filtered to {len(DATASETS)} datasets ({', '.join(DATASETS)}): "
                     f"{len(mention_df):,} mentions")

    # Exclude HyperCoref sport topics
    if EXCLUDE_TOPICS and HYPERCOREF_EXCLUDE_TOPICS:
        mask = (mention_df["dataset"] == "HyperCoref") & \
               (mention_df["topic"].isin(HYPERCOREF_EXCLUDE_TOPICS))
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        if is_main:
            logger.info(f"Excluded {n_excluded:,} HyperCoref mentions with topics "
                        f"{HYPERCOREF_EXCLUDE_TOPICS}")

    # Exclude WECEng topics
    if EXCLUDE_TOPICS and WECENG_EXCLUDE_TOPICS:
        mask = (mention_df["dataset"] == "WECEng") & \
               (mention_df["topic"].isin(WECENG_EXCLUDE_TOPICS))
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        if is_main:
            logger.info(f"Excluded {n_excluded:,} WECEng mentions with topics "
                        f"{WECENG_EXCLUDE_TOPICS}")

    # Determine training datasets based on mode
    if TRAIN_MODE == "single":
        train_datasets = [SINGLE_TRAIN_DATASET]
    elif TRAIN_MODE == "hypercoref_main":
        train_datasets = ["HyperCoref"]
    else:  # mix
        train_datasets = list(DATASETS)

    # Build a training-only dataframe (all mentions, but only train-datasets for the train split)
    train_mention_df = mention_df.copy()
    train_mask = train_mention_df["split"] == "train"
    dataset_mask = train_mention_df["dataset"].isin(train_datasets)
    # Keep: train split rows only if in train_datasets; val/test rows from all DATASETS
    train_mention_df = train_mention_df[~train_mask | (train_mask & dataset_mask)].copy()

    if is_main:
        logger.info(f"Train mode: {TRAIN_MODE}  |  Training datasets: {train_datasets}")
        n_train = (train_mention_df["split"] == "train").sum()
        logger.info(f"  Training mentions: {n_train:,}")

    # --- Tokenizer ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])

    # --- Build training dataloader ---
    per_gpu_batch = args.batch_size // args.world_size
    train_dataloader, chain_index = build_dataloader(
        mention_df=train_mention_df,
        tokenizer=tokenizer,
        batch_size=per_gpu_batch if is_distributed else args.batch_size,
        max_length=args.max_length,
        tau=args.tau,
        gamma=args.gamma,
        singleton_frac=args.singleton_frac,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if is_main:
        logger.info(f"Training dataloader: {len(train_dataloader)} batches/epoch")

    # --- Build per-dataset validation dataloaders ---
    # In single mode, evaluate only on the training dataset's own val split.
    val_datasets = [SINGLE_TRAIN_DATASET] if TRAIN_MODE == "single" else None
    val_resources = build_val_resources(
        mention_df, tokenizer, args.max_length, args.val_batch_size, args.num_workers,
        datasets=val_datasets,
    )
    if is_main:
        for ds_name, res in sorted(val_resources.items()):
            logger.info(f"  Val {ds_name:20s}: {res['n_mentions']:>6,} mentions  "
                        f"{res['n_subtopics']:>4} subtopics")
        if TRAIN_MODE == "single":
            logger.info(f"  Monitor metric: pd_LEA F1 on {SINGLE_TRAIN_DATASET} val")
        else:
            logger.info(f"  Monitor metric: avg pd_LEA F1 across all {len(val_resources)} datasets")
        logger.info(f"  Clustering: per-subtopic (matching CDCR evaluation protocol)")

    # --- Model ---
    model = CorefEncoder(tokenizer, mlp_hidden_dim=args.mlp_dim)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {n_params / 1e6:.1f}M params ({n_trainable / 1e6:.1f}M trainable)")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup + cosine decay
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Loss ---
    ms_loss_fn = MultiSimilarityLoss(
        alpha=args.ms_alpha,
        beta=args.ms_beta,
        lam=args.ms_lambda,
    )

    # --- Label encoder ---
    label_encoder = LabelEncoder()

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # --- Run directory (suffix separates runs cleanly) ---
    run_dir = Path(args.output_dir) / args.run_suffix
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        # File logger — write to /tmp (volume fs doesn't support flush), copy to run_dir at end
        local_log_path = Path("/tmp") / f"ms_coref_{args.run_suffix}_train.log"
        fh = logging.FileHandler(local_log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logging.getLogger().addHandler(fh)

        # Save all hyperparameters + config to JSON
        params = {
            "run_suffix": args.run_suffix,
            "train_mode": TRAIN_MODE,
            "train_datasets": train_datasets,
            "eval_datasets": sorted(val_resources.keys()),
            "eval_threshold": EVAL_THRESHOLD,
            **vars(args),
        }
        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2, default=str)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Params saved to {run_dir / 'params.json'}")

    # Checkpoint selection tracking
    if TRAIN_MODE == "single":
        best_pd_lea = 0.0
    else:
        # One checkpoint selected by avg pd_LEA F1 across all datasets
        best_val_f1 = 0.0

    # Per-epoch metrics accumulated in memory, written fully each epoch
    # (avoids Illegal seek on Databricks volume filesystems which don't support append mode)
    metrics_log_path = run_dir / "metrics_log.jsonl"
    metrics_log_rows = []

    # --- Training ---
    if is_main:
        logger.info(f"Starting training: {args.num_epochs} epochs, "
                     f"batch_size={args.batch_size}, lr={args.lr}")

    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_chain_losses = defaultdict(list)
        epoch_mined_pos = 0
        epoch_mined_neg = 0
        epoch_valid_anchors = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            str_labels = batch["label"]  # list of string chain IDs

            # Encode labels to integers
            int_labels = label_encoder.encode(str_labels).to(device)

            B = input_ids.size(0)
            mb = args.micro_batch_size

            # --- Phase 1: Forward all micro-batches WITHOUT grad ---
            # Collect embeddings (small: B x 2048). RoBERTa activations
            # are freed after each chunk since there's no grad graph.
            # Bypass DDP so the reducer isn't triggered on these forwards.
            no_grad_model = model.module if is_distributed else model
            emb_chunks = []
            with torch.no_grad():
                for start in range(0, B, mb):
                    end = min(start + mb, B)
                    with torch.amp.autocast("cuda", enabled=args.fp16):
                        chunk_emb = no_grad_model(
                            input_ids[start:end], attention_mask[start:end]
                        )
                    emb_chunks.append(chunk_emb)

            all_emb = torch.cat(emb_chunks, dim=0)  # (B, D) detached

            # --- Phase 2: Compute MS Loss on full batch, backprop to embeddings only ---
            all_emb.requires_grad_(True)
            if is_distributed:
                all_emb_gathered, all_labels = gather_embeddings(all_emb, int_labels)
            else:
                all_emb_gathered, all_labels = all_emb, int_labels

            with torch.amp.autocast("cuda", enabled=args.fp16):
                loss, per_label_losses, pair_stats = ms_loss_fn(all_emb_gathered, all_labels)

            scaler.scale(loss).backward()
            # emb_grad: gradient of loss w.r.t. local embeddings (B, D)
            emb_grad = all_emb.grad.detach().clone()

            # --- Phase 3: Re-forward each micro-batch WITH grad ---
            # Use pre-computed emb_grad to drive backward through the model.
            # Only one micro-batch's activations are in memory at a time.
            # Suppress DDP all-reduce until the last micro-batch so gradients
            # accumulate locally first, then sync once per optimizer step.
            optimizer.zero_grad()
            micro_starts = list(range(0, B, mb))
            for i, start in enumerate(micro_starts):
                end = min(start + mb, B)
                is_last = (i == len(micro_starts) - 1)
                sync_ctx = (contextlib.nullcontext()
                            if (not is_distributed or is_last)
                            else model.no_sync())
                with sync_ctx:
                    with torch.amp.autocast("cuda", enabled=args.fp16):
                        chunk_emb = model(
                            input_ids[start:end], attention_mask[start:end]
                        )
                    # Equivalent to: (d_loss/d_emb) * (d_emb/d_theta)
                    chunk_emb.backward(emb_grad[start:end])

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            # Accumulate per-chain losses for difficulty EMA update at epoch end
            for label_val, losses in per_label_losses.items():
                epoch_chain_losses[label_val].extend(losses)

            epoch_mined_pos += pair_stats["n_mined_pos"]
            epoch_mined_neg += pair_stats["n_mined_neg"]
            epoch_valid_anchors += pair_stats["n_valid_anchors"]

            # Logging
            if is_main and global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}  "
                    f"Step {batch_idx+1}/{len(train_dataloader)}  "
                    f"Loss: {loss_val:.4f}  "
                    f"LR: {lr:.2e}"
                )

        # --- End of epoch ---
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)

        if is_main:
            avg_pos = epoch_mined_pos / max(epoch_steps, 1)
            avg_neg = epoch_mined_neg / max(epoch_steps, 1)
            avg_anchors = epoch_valid_anchors / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, "
                f"time={epoch_time:.0f}s  |  "
                f"avg mined pairs/batch: pos={avg_pos:.1f}  neg={avg_neg:.1f}  "
                f"valid_anchors={avg_anchors:.1f}"
            )

        # Update chain difficulty EMA
        chain_loss_means = {
            cid: np.mean(losses)
            for cid, losses in epoch_chain_losses.items()
            if losses
        }
        chain_index.update_difficulty(chain_loss_means, alpha=args.difficulty_alpha)

        if is_main and len(chain_loss_means) > 0:
            vals = list(chain_loss_means.values())
            logger.info(
                f"  Difficulty update: {len(vals)} chains, "
                f"mean_loss={np.mean(vals):.4f}, "
                f"max_loss={np.max(vals):.4f}"
            )

        # --- Validation at fixed threshold ---
        if is_main and (epoch + 1) % args.val_every == 0:
            logger.info(f"Evaluating at fixed threshold {EVAL_THRESHOLD}...")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raw_model = model.module if is_distributed else model

            # Compute embeddings once per dataset, cluster per subtopic
            dataset_names = sorted(val_resources.keys())
            eval_metrics = {}
            for ds_name, res in val_resources.items():
                embs, labels = compute_embeddings(raw_model, res["dataloader"], device)
                metrics = evaluate_per_subtopic(
                    embs, labels, res["pd_tuples"], res["subtopic_ids"],
                    threshold=EVAL_THRESHOLD,
                )
                eval_metrics[ds_name] = metrics

            if device.type == "cuda":
                torch.cuda.empty_cache()

            avg_pd_lea_f1 = np.mean([eval_metrics[ds]["pd_lea_f1"] for ds in dataset_names])
            avg_conll_f1 = np.mean([eval_metrics[ds]["conll_f1"] for ds in dataset_names])

            # Log per-dataset results
            for ds_name in dataset_names:
                m = eval_metrics[ds_name]
                logger.info(
                    f"  {ds_name:20s}  "
                    f"CoNLL={m['conll_f1']:.4f}  "
                    f"MUC={m['muc_f1']:.4f}  B3={m['b3_f1']:.4f}  CEAFe={m['ceafe_f1']:.4f}  "
                    f"LEA={m['lea_f1']:.4f}  "
                    f"pd_LEA={m['pd_lea_f1']:.4f} (P={m['pd_lea_precision']:.4f} R={m['pd_lea_recall']:.4f})  "
                    f"clusters: gold={m['n_gold_clusters']} pred={m['n_pred_clusters']}"
                )
            if TRAIN_MODE == "single":
                logger.info(f"  pd_LEA F1 = {avg_pd_lea_f1:.4f}  CoNLL F1 = {avg_conll_f1:.4f}  (threshold={EVAL_THRESHOLD})")
            else:
                logger.info(f"  Avg pd_LEA F1 = {avg_pd_lea_f1:.4f}  Avg CoNLL F1 = {avg_conll_f1:.4f}  (threshold={EVAL_THRESHOLD})")

            # Shared checkpoint payload (reused for all saves this epoch)
            ckpt_payload = {
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_threshold": EVAL_THRESHOLD,
                "train_mode": TRAIN_MODE,
                "train_datasets": train_datasets,
                "all_metrics": {
                    ds: {k: float(v) for k, v in m.items()}
                    for ds, m in eval_metrics.items()
                },
                "args": vars(args),
            }

            if TRAIN_MODE == "single":
                # Select checkpoints based on SINGLE_TRAIN_DATASET val split only
                sel_m = eval_metrics[SINGLE_TRAIN_DATASET]
                sel_pd_lea = sel_m["pd_lea_f1"]

                if sel_pd_lea > best_pd_lea:
                    best_pd_lea = sel_pd_lea
                    _safe_torch_save({**ckpt_payload, "selection_metric": "pd_lea_f1",
                                      "selection_dataset": SINGLE_TRAIN_DATASET,
                                      "selection_value": sel_pd_lea},
                                     ckpt_dir / "best_model_pd_lea.pt")
                    _safe_torch_save({"epoch": epoch + 1, "model_state_dict": raw_model.state_dict(),
                                      "train_mode": TRAIN_MODE, "train_datasets": train_datasets,
                                      "args": vars(args)},
                                     ckpt_dir / "best_model_pd_lea_weights.pt")
                    logger.info(f"  New best (pd_LEA): {sel_pd_lea:.4f} on {SINGLE_TRAIN_DATASET} val")
            else:
                # mix / hypercoref_main: select by avg pd_LEA across all datasets
                if avg_pd_lea_f1 > best_val_f1:
                    best_val_f1 = avg_pd_lea_f1
                    _safe_torch_save({**ckpt_payload, "best_avg_pd_lea_f1": best_val_f1},
                                     ckpt_dir / "best_model.pt")
                    _safe_torch_save({"epoch": epoch + 1, "model_state_dict": raw_model.state_dict(),
                                      "train_mode": TRAIN_MODE, "train_datasets": train_datasets,
                                      "args": vars(args)},
                                     ckpt_dir / "best_model_weights.pt")
                    logger.info(f"  New best (avg pd_LEA): {best_val_f1:.4f}")

            # Write metrics log — full overwrite each epoch (volume fs doesn't support append)
            metrics_log_rows.append({
                "epoch": epoch + 1,
                "avg_pd_lea_f1": float(avg_pd_lea_f1),
                "avg_conll_f1": float(avg_conll_f1),
                "per_dataset": {
                    ds: {k: float(v) for k, v in m.items()}
                    for ds, m in eval_metrics.items()
                },
            })
            with open(metrics_log_path, "w") as f:
                for row in metrics_log_rows:
                    f.write(json.dumps(row) + "\n")

        # Save periodic checkpoint (model-only to stay within filesystem limits)
        if is_main and (epoch + 1) % args.save_every == 0:
            raw_model = model.module if is_distributed else model
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
            _safe_torch_save({
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "train_mode": TRAIN_MODE,
                "train_datasets": train_datasets,
                "args": vars(args),
            }, ckpt_path)

    if is_main:
        if TRAIN_MODE == "single":
            logger.info(f"Training complete. Best pd_LEA={best_pd_lea:.4f}  (on {SINGLE_TRAIN_DATASET} val)")
        else:
            logger.info(f"Training complete. Best avg pd_LEA F1: {best_val_f1:.4f}")

    if is_distributed:
        dist.destroy_process_group()

    # Copy train log from /tmp to volume now that training is complete
    if is_main:
        try:
            fh.close()
            logging.getLogger().removeHandler(fh)
            dest_log = run_dir / "train.log"
            with open(local_log_path, "rb") as fsrc, open(dest_log, "wb") as fdst:
                while buf := fsrc.read(64 * 1024 * 1024):
                    fdst.write(buf)
            logger.info(f"Train log saved to {dest_log}")
        except Exception as e:
            logger.warning(f"Could not copy train log to volume: {e} (kept at {local_log_path})")

    # --- Post-training inference on test set ---
    # Only run on main process, after distributed teardown.
    # Uses best_model_pd_lea.pt (single mode) or best_model.pt (mix/hypercoref_main).
    if is_main:
        if TRAIN_MODE == "single":
            ckpt_for_eval = ckpt_dir / "best_model_pd_lea.pt"
        else:
            ckpt_for_eval = ckpt_dir / "best_model.pt"

        if ckpt_for_eval.exists():
            logger.info(f"\nRunning conll_eval.py on test set with {ckpt_for_eval.name} ...")
            conll_eval_script = Path(os.getcwd()) / "conll_eval.py"
            cmd = [
                sys.executable, str(conll_eval_script),
                "--checkpoint", str(ckpt_for_eval),
                "--data-dir", args.data_dir,
                "--output-dir", str(run_dir / "conll_output"),
                "--split", "test",
                "--setting", args.run_suffix,
                "--max-length", str(args.max_length),
                "--mlp-dim", str(args.mlp_dim),
                "--batch-size", str(args.val_batch_size),
            ]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                logger.warning(f"conll_eval.py exited with code {result.returncode}")
            else:
                logger.info("conll_eval.py completed successfully.")
        else:
            logger.warning(f"No checkpoint found at {ckpt_for_eval}, skipping conll_eval.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config():
    base = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    cfg = _load_config()
    p = argparse.ArgumentParser(description="MS Loss coreference training")

    # Data / run identity
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default=cfg.get("output_dir"))
    p.add_argument("--run-suffix", type=str, default=None,
                   help="Subdirectory name for this run (default: auto from TRAIN_MODE/dataset)")
    p.add_argument("--max-length", type=int, default=128)

    # PK Sampling
    p.add_argument("--batch-size", type=int, default=2048,
                    help="Total batch size across all GPUs")
    p.add_argument("--micro-batch-size", type=int, default=128,
                    help="Forward pass chunk size per GPU (controls GPU memory)")
    p.add_argument("--tau", type=float, default=1.0,
                    help="PD exponent for chain weighting")
    p.add_argument("--gamma", type=float, default=0.5,
                    help="Difficulty EMA exponent for chain weighting")
    p.add_argument("--singleton-frac", type=float, default=0.20)
    p.add_argument("--difficulty-alpha", type=float, default=0.1,
                    help="EMA smoothing for difficulty updates")

    # Model
    p.add_argument("--mlp-dim", type=int, default=1024)

    # MS Loss
    p.add_argument("--ms-alpha", type=float, default=2.0)
    p.add_argument("--ms-beta", type=float, default=50.0)
    p.add_argument("--ms-lambda", type=float, default=0.5)

    # Training
    p.add_argument("--num-epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_false", dest="fp16")

    # Validation
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--val-every", type=int, default=1,
                    help="Validate every N epochs")

    # Checkpointing
    p.add_argument("--save-every", type=int, default=5)

    # System
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--world-size", type=int, default=1,
                    help="Number of GPUs (auto-detected with torchrun)")

    args, _ = p.parse_known_args()

    # Auto-detect world size from torchrun / TorchDistributor
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Auto-derive run suffix from training config if not provided
    if args.run_suffix is None:
        all_six = ["HyperCoref", "NP4E", "NewsWCL50", "NIdent", "ECBplus", "ECBplusMETAm"]
        if TRAIN_MODE == "single":
            args.run_suffix = f"single_{SINGLE_TRAIN_DATASET}"
        elif TRAIN_MODE == "hypercoref_main":
            args.run_suffix = "hypercoref_main"
        elif sorted(DATASETS) == sorted(all_six):
            args.run_suffix = "mix"
        else:
            # Subset mix — encode the dataset names to avoid collision
            ds_tag = "_".join(sorted(DATASETS))
            args.run_suffix = f"mix_{ds_tag}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
