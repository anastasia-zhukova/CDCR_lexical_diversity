"""
Pluggable training core shared by every loss/model regime.

This is the key integration point between the two historical codebases:
  - ms-coref's approach:      MultiSimilarityLoss + PK batch sampler (dynamic,
                               chain-labeled batches; sampler.py/build_dataloader).
  - cl-coref-orig's approach: ContrastiveLoss + static pair sampler (SBERT
                               MST positives + KNN hard negatives; pair_sampler.py).

Both regimes share the same encoder architecture (coref_encoder.CorefEncoder)
and the same evaluation protocol (coref_encoder.evaluate_per_subtopic) — only
the loss function, the data/batch construction (build_model_and_data), and
the epoch loop (run_train_epoch) differ across models. That's the piece that
changes when adding a new training regime.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coref_encoder import CorefEncoder
from pair_sampler import sample_pairs_from_mentions
from sampler import PairDataset, build_dataloader, pair_collate_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Losses — this is where models diverge
# ---------------------------------------------------------------------------


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss from Wang et al. (CVPR 2019). Specific to the
    ms-coref (PK batch sampler) training regime.

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
            Scalar loss, per-anchor loss dict keyed by label (for difficulty tracking),
            pair-mining stats dict.
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

        pair_stats = {
            "n_mined_pos": n_mined_pos,
            "n_mined_neg": n_mined_neg,
            "n_valid_anchors": n_valid,
        }
        return loss, per_label_loss, pair_stats


class ContrastiveLoss(nn.Module):
    """
    Margin-based contrastive loss, ported from cl-coref-orig's
    CustomContrastiveLoss (model.py). Specific to the pair-sampled training
    regime (pair_sampler.py + sampler.PairDataset), as an alternative to
    MultiSimilarityLoss.

    loss = label * distance^2 + (1 - label) * relu(margin - distance)^2
    where distance = 1 - cosine_similarity(u, v).
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin  # paper: 0.4 for events, 0.7 for entities

    def forward(self, u, v, label):
        distance = 1 - F.cosine_similarity(u, v)
        loss = label.float() * distance.pow(2) + (1 - label).float() * F.relu(
            self.margin - distance
        ).pow(2)
        return loss.mean()


# ---------------------------------------------------------------------------
# Cross-GPU embedding gathering (MS Loss regime only — needs the whole batch
# for online positive/negative mining across GPUs)
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
    gathered_labels = [
        torch.zeros_like(labels_int) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_labels, labels_int)
    all_labels = torch.cat(gathered_labels, dim=0)

    return all_embeddings, all_labels


class LabelEncoder:
    """Maps string coref_chain IDs to contiguous integer labels (MS Loss regime)."""

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
# load_and_filter_mentions — unify the training data across regimes
# ---------------------------------------------------------------------------


def load_and_filter_mentions(
    data_dir,
    datasets,
    train_mode,
    single_train_dataset=None,
    hypercoref_exclude_topics=None,
    is_main=True,
):
    """
    Load the processed uCDCR parquet and build the mention tables shared by
    every training regime.

    Args:
        data_dir: directory containing ucdcr_mentions_processed.parquet.
        datasets: subset of uCDCR datasets to use for this experiment.
        train_mode: "single" / "hypercoref_main" / "mix".
        single_train_dataset: required when train_mode == "single".
        hypercoref_exclude_topics: HyperCoref topics to drop (e.g. {"Sport"}).
        is_main: whether to emit info logs (for DDP main-process gating).

    Returns:
        mention_df: filtered mentions across all `datasets` (all splits) —
            used for validation (always evaluated on every dataset).
        train_mention_df: same rows, but the "train" split is restricted to
            train_datasets (val/test rows are kept for all datasets).
        train_datasets: list of dataset names actually used for training.
    """
    data_path = Path(data_dir) / "ucdcr_mentions_processed.parquet"
    mention_df = pd.read_parquet(data_path)
    if is_main:
        logger.info(f"Loaded {len(mention_df):,} mentions from {data_path}")

    mention_df = mention_df[mention_df["dataset"].isin(datasets)].copy()
    if is_main:
        logger.info(
            f"Filtered to {len(datasets)} datasets ({', '.join(datasets)}): "
            f"{len(mention_df):,} mentions"
        )

    if hypercoref_exclude_topics:
        mask = (mention_df["dataset"] == "HyperCoref") & (
            mention_df["topic"].isin(hypercoref_exclude_topics)
        )
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        if is_main:
            logger.info(
                f"Excluded {n_excluded:,} HyperCoref mentions with topics "
                f"{hypercoref_exclude_topics}"
            )

    if train_mode == "single":
        if not single_train_dataset:
            raise ValueError(
                "single_train_dataset is required when train_mode == 'single'"
            )
        train_datasets = [single_train_dataset]
    elif train_mode == "hypercoref_main":
        train_datasets = ["HyperCoref"]
    else:  # mix
        train_datasets = list(datasets)

    train_mention_df = mention_df.copy()
    train_mask = train_mention_df["split"] == "train"
    dataset_mask = train_mention_df["dataset"].isin(train_datasets)
    # Keep: train split rows only if in train_datasets; val/test rows from all datasets
    train_mention_df = train_mention_df[
        ~train_mask | (train_mask & dataset_mask)
    ].copy()

    if is_main:
        logger.info(f"Train mode: {train_mode}  |  Training datasets: {train_datasets}")
        n_train = (train_mention_df["split"] == "train").sum()
        logger.info(f"  Training mentions: {n_train:,}")

    return mention_df, train_mention_df, train_datasets


# ---------------------------------------------------------------------------
# build_model_and_data — chooses which architecture/dataloader/loss to use
# ---------------------------------------------------------------------------


def build_model_and_data(
    training_mode, train_mention_df, tokenizer, args, is_distributed, is_main
):
    """
    Build the shared CorefEncoder plus the training dataloader and loss
    appropriate for `training_mode`.

    The encoder architecture itself never changes across regimes (it's
    identical to cl-coref-orig's SiameseNetwork) — only how batches are
    constructed and how the loss is computed does.

    Args:
        training_mode: "ms_loss" (PK batch sampler) or "contrastive" (pair sampler).
        train_mention_df: training-split mention table (see load_and_filter_mentions).
        tokenizer: RoBERTa tokenizer with [E]/[/E] added.
        args: parsed CLI args (see train.py's parse_args).
        is_distributed: whether DDP is active.
        is_main: whether to emit info logs.

    Returns:
        model, train_dataloader, loss_fn, extra
            extra: {"chain_index": ChainIndex, "label_encoder": LabelEncoder}
            for "ms_loss"; {"chain_index": None, "label_encoder": None} for
            "contrastive".
    """
    model = CorefEncoder(tokenizer, mlp_hidden_dim=args.mlp_dim)

    if training_mode == "ms_loss":
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
        loss_fn = MultiSimilarityLoss(
            alpha=args.ms_alpha, beta=args.ms_beta, lam=args.ms_lambda
        )
        extra = {"chain_index": chain_index, "label_encoder": LabelEncoder()}

    elif training_mode == "contrastive":
        train_split_df = train_mention_df[train_mention_df["split"] == "train"].copy()
        pairs_df = sample_pairs_from_mentions(
            train_split_df,
            params={
                "split": "train",
                "sbert_model": args.contrastive_sbert_model,
                "k_hard_negatives": args.contrastive_k_negatives,
                "random_seed": args.seed,
            },
        )
        if is_main:
            pos_frac = pairs_df["coreference"].mean() if len(pairs_df) else 0.0
            logger.info(
                f"Sampled {len(pairs_df):,} contrastive pairs ({pos_frac * 100:.1f}% positive)"
            )

        pair_dataset = PairDataset(
            train_split_df, pairs_df, tokenizer, max_length=args.max_length
        )
        train_dataloader = DataLoader(
            pair_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
            num_workers=args.num_workers,
        )
        loss_fn = ContrastiveLoss(margin=args.contrastive_margin)
        extra = {"chain_index": None, "label_encoder": None}

    else:
        raise ValueError(f"Unknown training_mode: {training_mode!r}")

    return model, train_dataloader, loss_fn, extra


# ---------------------------------------------------------------------------
# run_train_epoch — this is what changes across models
# ---------------------------------------------------------------------------


def run_train_epoch(
    training_mode,
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    epoch,
    global_step,
    label_encoder=None,
    is_distributed=False,
    is_main=True,
):
    """
    Run one training epoch and return (stats, global_step).

    stats always contains: avg_loss, epoch_time, chain_loss_means (per-chain
    MS Loss contributions for difficulty EMA — empty for contrastive),
    avg_mined_pos / avg_mined_neg / avg_valid_anchors (None for contrastive).
    """
    if training_mode == "ms_loss":
        return _run_ms_loss_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            global_step,
            label_encoder,
            is_distributed,
            is_main,
        )
    if training_mode == "contrastive":
        return _run_contrastive_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            global_step,
            is_main,
        )
    raise ValueError(f"Unknown training_mode: {training_mode!r}")


def _run_ms_loss_epoch(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    epoch,
    global_step,
    label_encoder,
    is_distributed,
    is_main,
):
    """MS Loss regime: PK-sampled chain batches, memory-efficient 3-phase
    chunked forward/backward (the loss needs the whole batch for online
    positive/negative mining, but activations for the whole batch don't fit)."""
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

        int_labels = label_encoder.encode(str_labels).to(device)

        B = input_ids.size(0)
        mb = args.micro_batch_size

        # --- Phase 1: Forward all micro-batches WITHOUT grad ---
        emb_chunks = []
        with torch.no_grad():
            for start in range(0, B, mb):
                end = min(start + mb, B)
                with torch.amp.autocast("cuda", enabled=args.fp16):
                    chunk_emb = model(input_ids[start:end], attention_mask[start:end])
                emb_chunks.append(chunk_emb)

        all_emb = torch.cat(emb_chunks, dim=0)  # (B, D) detached

        # --- Phase 2: Compute MS Loss on full batch, backprop to embeddings only ---
        all_emb.requires_grad_(True)
        if is_distributed:
            all_emb_gathered, all_labels = gather_embeddings(all_emb, int_labels)
        else:
            all_emb_gathered, all_labels = all_emb, int_labels

        with torch.amp.autocast("cuda", enabled=args.fp16):
            loss, per_label_losses, pair_stats = loss_fn(all_emb_gathered, all_labels)

        scaler.scale(loss).backward()
        emb_grad = all_emb.grad.detach().clone()

        # --- Phase 3: Re-forward each micro-batch WITH grad ---
        optimizer.zero_grad()
        for start in range(0, B, mb):
            end = min(start + mb, B)
            with torch.amp.autocast("cuda", enabled=args.fp16):
                chunk_emb = model(input_ids[start:end], attention_mask[start:end])
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

        for label_val, losses in per_label_losses.items():
            epoch_chain_losses[label_val].extend(losses)

        epoch_mined_pos += pair_stats["n_mined_pos"]
        epoch_mined_neg += pair_stats["n_mined_neg"]
        epoch_valid_anchors += pair_stats["n_valid_anchors"]

        if is_main and global_step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}  "
                f"Step {batch_idx+1}/{len(train_dataloader)}  "
                f"Loss: {loss_val:.4f}  "
                f"LR: {lr:.2e}"
            )

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
    else:
        avg_pos = avg_neg = avg_anchors = None

    chain_loss_means = {
        cid: np.mean(losses) for cid, losses in epoch_chain_losses.items() if losses
    }

    stats = {
        "avg_loss": avg_loss,
        "epoch_time": epoch_time,
        "chain_loss_means": chain_loss_means,
        "avg_mined_pos": avg_pos,
        "avg_mined_neg": avg_neg,
        "avg_valid_anchors": avg_anchors,
    }
    return stats, global_step


def _run_contrastive_epoch(
    model,
    train_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    epoch,
    global_step,
    is_main,
):
    """Contrastive-loss regime: static mention pairs, ordinary forward/backward
    per batch (no online mining, so no need for the chunked replay trick)."""
    model.train()
    epoch_loss = 0.0
    epoch_steps = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids1, attention_mask1, input_ids2, attention_mask2, labels = batch
        input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)
        input_ids2, attention_mask2 = input_ids2.to(device), attention_mask2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.fp16):
            embed1 = model(input_ids1, attention_mask1)
            embed2 = model(input_ids2, attention_mask2)
            loss = loss_fn(embed1, embed2, labels)

        scaler.scale(loss).backward()
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

        if is_main and global_step % args.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}  "
                f"Step {batch_idx+1}/{len(train_dataloader)}  "
                f"Loss: {loss_val:.4f}  "
                f"LR: {lr:.2e}"
            )

    epoch_time = time.time() - t0
    avg_loss = epoch_loss / max(epoch_steps, 1)
    if is_main:
        logger.info(
            f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.0f}s"
        )

    stats = {
        "avg_loss": avg_loss,
        "epoch_time": epoch_time,
        "chain_loss_means": {},
        "avg_mined_pos": None,
        "avg_mined_neg": None,
        "avg_valid_anchors": None,
    }
    return stats, global_step
