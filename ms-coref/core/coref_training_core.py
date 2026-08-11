"""
Shared core logic for MS Loss cross-document coreference training.

This module holds the training/eval building blocks that are common to both
`train.py` (full DDP training run) and `ray_tune_sweep.py` (single-GPU Ray
Tune HPO trials), so neither script has to import "main-script" internals
from the other. It has no `__main__` entry point of its own.

This module never reads config.yaml or hardcodes experiment configuration
(datasets, train mode, topic exclusions, eval threshold, etc.) itself —
those values are the caller's responsibility (`train.py` / `ray_tune_sweep.py`
read config.yaml and pass the resulting values in as function arguments), so
this "auxiliary" module stays independent of any one script's configuration.

Contents:
  - MultiSimilarityLoss, LabelEncoder
  - Cross-GPU embedding gathering (GatherWithGrad, gather_embeddings)
  - load_and_filter_mentions(): parquet load + dataset/topic filtering
  - build_val_resources(): per-dataset validation dataloaders/metadata
  - build_model_and_data(): tokenizer + train/val dataloaders + model
  - build_cosine_schedule(): linear warmup + cosine decay LR schedule
  - run_train_epoch(): the 3-phase memory-saving MS-loss training step
  - evaluate_validation(): per-dataset + averaged pd_LEA/CoNLL evaluation
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from .coref_utils import CorefEncoder, compute_embeddings, evaluate_per_subtopic
from sampler import MentionDataset, build_dataloader


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


# ---------------------------------------------------------------------------
# Data loading / filtering
# ---------------------------------------------------------------------------

def load_and_filter_mentions(
    data_dir,
    datasets,
    train_mode,
    single_train_dataset=None,
    exclude_topics=True,
    hypercoref_exclude_topics=None,
    weceng_exclude_topics=None,
    log=None,
):
    """
    Load the processed uCDCR mention parquet, filter to the configured
    `datasets`, apply per-dataset topic exclusions, and split off the
    training dataframe according to `train_mode`.

    All experiment-configuration values (which datasets, which train mode,
    topic exclusions, etc.) are supplied by the caller (e.g. `train.py` /
    `ray_tune_sweep.py`, which read them from config.yaml) rather than
    hardcoded here.

    Args:
        data_dir: directory containing ucdcr_mentions_processed.parquet.
        datasets: list of dataset names to keep (across all splits).
        train_mode: "single", "hypercoref_main", or "mix" — selects which
            datasets are used for the *train* split (val/test rows for all
            `datasets` are always retained regardless of this setting).
        single_train_dataset: dataset name to use when train_mode == "single".
            Required (and only used) in that mode.
        exclude_topics: if True, apply the per-dataset topic exclusion sets
            below. If False, no topic-based filtering is applied.
        hypercoref_exclude_topics: set/list of HyperCoref topics to drop
            (only applied when exclude_topics=True). None/empty = no-op.
        weceng_exclude_topics: set/list of WECEng topics to drop (only
            applied when exclude_topics=True). None/empty = no-op.
        log: optional callable(str) for progress logging (e.g. logger.info).
             Pass None to run silently (e.g. inside a Ray Tune trial).

    Returns:
        mention_df: filtered dataframe (all splits, all configured datasets).
        train_mention_df: dataframe with the train split restricted to
            train_datasets; val/test rows for all datasets are retained.
        train_datasets: list of dataset names used for the train split
            (depends on train_mode).
    """
    data_path = Path(data_dir) / "ucdcr_mentions_processed.parquet"
    mention_df = pd.read_parquet(data_path)
    if log:
        log(f"Loaded {len(mention_df):,} mentions from {data_path}")

    mention_df = mention_df[mention_df["dataset"].isin(datasets)].copy()
    if log:
        log(f"Filtered to {len(datasets)} datasets ({', '.join(datasets)}): "
            f"{len(mention_df):,} mentions")

    if exclude_topics and hypercoref_exclude_topics:
        mask = (mention_df["dataset"] == "HyperCoref") & \
               mention_df["topic"].isin(hypercoref_exclude_topics)
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        if log:
            log(f"Excluded {n_excluded:,} HyperCoref mentions with topics "
                f"{hypercoref_exclude_topics}")

    if exclude_topics and weceng_exclude_topics:
        mask = (mention_df["dataset"] == "WECEng") & \
               mention_df["topic"].isin(weceng_exclude_topics)
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        if log:
            log(f"Excluded {n_excluded:,} WECEng mentions with topics "
                f"{weceng_exclude_topics}")

    if train_mode == "single":
        train_datasets = [single_train_dataset]
    elif train_mode == "hypercoref_main":
        train_datasets = ["HyperCoref"]
    else:  # mix
        train_datasets = list(datasets)

    # Keep: train split rows only if in train_datasets; val/test rows from all datasets
    train_mention_df = mention_df.copy()
    train_mask = train_mention_df["split"] == "train"
    dataset_mask = train_mention_df["dataset"].isin(train_datasets)
    train_mention_df = train_mention_df[~train_mask | (train_mask & dataset_mask)].copy()

    if log:
        log(f"Train mode: {train_mode}  |  Training datasets: {train_datasets}")
        n_train = (train_mention_df["split"] == "train").sum()
        log(f"  Training mentions: {n_train:,}")

    return mention_df, train_mention_df, train_datasets


# ---------------------------------------------------------------------------
# Validation resources
# ---------------------------------------------------------------------------

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
# Tokenizer, dataloaders, and model construction
# ---------------------------------------------------------------------------

def build_model_and_data(
    mention_df,
    train_mention_df,
    *,
    batch_size,
    max_length,
    tau,
    gamma,
    singleton_frac,
    num_workers,
    seed,
    val_batch_size,
    mlp_dim,
    device,
    val_datasets=None,
    wrap_model_fn=None,
    log=None,
):
    """
    Build the tokenizer, training/validation dataloaders, and model — the
    setup sequence shared by train.py and ray_tune_sweep.py:
    tokenizer -> train dataloader -> validation resources -> model.

    Args:
        mention_df: full filtered dataframe (all splits), used to build
            validation resources.
        train_mention_df: dataframe with the train split restricted to the
            configured training datasets (see load_and_filter_mentions()).
        batch_size: training dataloader batch size (already divided across
            GPUs by the caller if running distributed).
        max_length: tokenizer max sequence length.
        tau, gamma, singleton_frac, num_workers, seed: passed to
            sampler.build_dataloader() for PK batch sampling.
        val_batch_size: validation dataloader batch size.
        mlp_dim: CorefEncoder MLP hidden dimension.
        device: torch device to move the model to.
        val_datasets: optional list of dataset names to restrict validation
            to (e.g. [SINGLE_TRAIN_DATASET] in "single" mode). None = all.
        wrap_model_fn: optional callable(model) -> model, applied right
            after the model is moved to device (e.g. DDP wrapping in
            train.py). None = model is used as-is (single-GPU sweeps).
        log: optional callable(str) for progress logging. None = silent.

    Returns:
        tokenizer, train_dataloader, chain_index, val_resources, model
    """
    # --- Tokenizer ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])

    # --- Training dataloader ---
    train_dataloader, chain_index = build_dataloader(
        mention_df=train_mention_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        tau=tau,
        gamma=gamma,
        singleton_frac=singleton_frac,
        num_workers=num_workers,
        seed=seed,
    )
    if log:
        log(f"Training dataloader: {len(train_dataloader)} batches/epoch")

    # --- Validation resources ---
    val_resources = build_val_resources(
        mention_df, tokenizer, max_length, val_batch_size, num_workers,
        datasets=val_datasets,
    )
    if log:
        for ds_name, res in sorted(val_resources.items()):
            log(f"  Val {ds_name:20s}: {res['n_mentions']:>6,} mentions  "
                f"{res['n_subtopics']:>4} subtopics")

    # --- Model ---
    model = CorefEncoder(tokenizer, mlp_hidden_dim=mlp_dim)
    model = model.to(device)
    if wrap_model_fn is not None:
        model = wrap_model_fn(model)

    if log:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"Model: {n_params / 1e6:.1f}M params ({n_trainable / 1e6:.1f}M trainable)")

    return tokenizer, train_dataloader, chain_index, val_resources, model


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_cosine_schedule(optimizer, total_steps, warmup_frac):
    """Linear warmup followed by cosine decay over total_steps."""
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop (shared 3-phase memory-saving MS-loss step)
# ---------------------------------------------------------------------------

def run_train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    ms_loss_fn,
    label_encoder,
    scaler,
    device,
    micro_batch_size,
    grad_clip=1.0,
    fp16=True,
    is_distributed=False,
    track_chain_losses=True,
    loss_in_fp32=False,
    step_callback=None,
):
    """
    Run one training epoch using the 3-phase memory-saving MS-loss step:

      1. No-grad forward in micro-batches to get embeddings (RoBERTa
         activations are freed after each chunk since there's no grad graph).
      2. Compute MS loss on the full batch of detached embeddings
         (optionally gathered across GPUs) and backward to obtain
         d(loss)/d(embeddings).
      3. Re-forward each micro-batch with grad and backprop the saved
         embedding gradients into the model — only one micro-batch's
         activations are in memory at a time.

    Args:
        model, dataloader, optimizer, scheduler, ms_loss_fn, label_encoder,
            scaler: standard training components for this epoch.
        device: torch device.
        micro_batch_size: forward-pass chunk size (controls GPU memory).
        grad_clip: max grad norm; <= 0 disables clipping.
        fp16: whether to use autocast/GradScaler mixed precision for the
            model forward passes.
        is_distributed: if True, gather embeddings/labels across GPUs
            before computing the loss.
        track_chain_losses: if True, accumulate per-chain-id losses for
            difficulty EMA updates (used by train.py). Ray Tune sweeps can
            skip this to keep trials lean.
        loss_in_fp32: if True, cast the gathered embeddings to float32 and
            disable autocast for the MS-loss computation (phase 2), for
            numerical stability. Used by the Ray Tune sweep.
        step_callback: optional fn(batch_idx, num_batches, loss_val) invoked
            after each optimizer step (e.g. for periodic logging).

    Returns:
        dict with keys: avg_loss, epoch_steps, chain_loss_means (empty dict
        if track_chain_losses=False), mined_pos, mined_neg, valid_anchors.
    """
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    epoch_chain_losses = defaultdict(list)
    epoch_mined_pos = 0
    epoch_mined_neg = 0
    epoch_valid_anchors = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        str_labels = batch["label"]  # list of string chain IDs
        int_labels = label_encoder.encode(str_labels).to(device)

        B = input_ids.size(0)
        mb = micro_batch_size

        # --- Phase 1: no-grad forward ---
        emb_chunks = []
        with torch.no_grad():
            for start in range(0, B, mb):
                end = min(start + mb, B)
                with torch.amp.autocast("cuda", enabled=fp16):
                    chunk_emb = model(input_ids[start:end], attention_mask[start:end])
                emb_chunks.append(chunk_emb)
        all_emb = torch.cat(emb_chunks, dim=0)  # (B, D) detached
        if loss_in_fp32:
            all_emb = all_emb.float()

        # --- Phase 2: MS loss on detached embeddings ---
        all_emb.requires_grad_(True)
        if is_distributed:
            all_emb_gathered, all_labels = gather_embeddings(all_emb, int_labels)
        else:
            all_emb_gathered, all_labels = all_emb, int_labels

        with torch.amp.autocast("cuda", enabled=(fp16 and not loss_in_fp32)):
            loss, per_label_losses, pair_stats = ms_loss_fn(all_emb_gathered, all_labels)

        scaler.scale(loss).backward()
        emb_grad = all_emb.grad.detach().clone()

        # --- Phase 3: re-forward with grad ---
        optimizer.zero_grad()
        for start in range(0, B, mb):
            end = min(start + mb, B)
            with torch.amp.autocast("cuda", enabled=fp16):
                chunk_emb = model(input_ids[start:end], attention_mask[start:end])
            chunk_emb.backward(emb_grad[start:end])

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        if not math.isnan(loss_val):
            epoch_loss += loss_val
            epoch_steps += 1

        if track_chain_losses:
            for label_val, losses in per_label_losses.items():
                epoch_chain_losses[label_val].extend(losses)

        epoch_mined_pos += pair_stats["n_mined_pos"]
        epoch_mined_neg += pair_stats["n_mined_neg"]
        epoch_valid_anchors += pair_stats["n_valid_anchors"]

        if step_callback is not None:
            step_callback(batch_idx, len(dataloader), loss_val)

    chain_loss_means = {
        cid: float(np.mean(losses))
        for cid, losses in epoch_chain_losses.items()
        if losses
    } if track_chain_losses else {}

    return {
        "avg_loss": epoch_loss / max(epoch_steps, 1),
        "epoch_steps": epoch_steps,
        "chain_loss_means": chain_loss_means,
        "mined_pos": epoch_mined_pos,
        "mined_neg": epoch_mined_neg,
        "valid_anchors": epoch_valid_anchors,
    }


# ---------------------------------------------------------------------------
# Validation evaluation
# ---------------------------------------------------------------------------

def evaluate_validation(model, val_resources, device, threshold):
    """
    Compute embeddings and per-subtopic clustering metrics for each
    validation dataset, plus averages across datasets.

    Args:
        model: the (unwrapped, non-DDP) model to evaluate.
        val_resources: dict from build_val_resources().
        device: torch device.
        threshold: cosine-distance clustering threshold. Supplied by the
            caller (e.g. from config.yaml's eval_threshold) rather than
            hardcoded here.

    Returns:
        eval_metrics: dict of dataset_name -> metrics dict (from
            evaluate_per_subtopic), e.g. pd_lea_f1, conll_f1, etc.
        avg_pd_lea_f1: float, mean pd_lea_f1 across datasets.
        avg_conll_f1: float, mean conll_f1 across datasets.
    """
    model.eval()
    dataset_names = sorted(val_resources.keys())
    eval_metrics = {}

    with torch.no_grad():
        for ds_name, res in val_resources.items():
            embs, labels = compute_embeddings(model, res["dataloader"], device)
            metrics = evaluate_per_subtopic(
                embs, labels, res["pd_tuples"], res["subtopic_ids"],
                threshold=threshold,
            )
            eval_metrics[ds_name] = metrics

    avg_pd_lea_f1 = float(np.mean([eval_metrics[ds]["pd_lea_f1"] for ds in dataset_names]))
    avg_conll_f1 = float(np.mean([eval_metrics[ds]["conll_f1"] for ds in dataset_names]))

    return eval_metrics, avg_pd_lea_f1, avg_conll_f1
