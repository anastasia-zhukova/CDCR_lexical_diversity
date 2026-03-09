"""
MS Loss training for cross-document coreference resolution.

Uses PK batch sampling with structured negative difficulty,
cross-GPU embedding gathering for large effective batches,
and pd_LEA validation on NewsWCL50.
"""

import argparse
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

from data_preparation import phrasing_diversity_calc
from pk_sampler import (
    ChainIndex,
    MentionDataset,
    PKBatchSampler,
    SingletonPool,
    build_pk_dataloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


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

        loss = torch.tensor(0.0, device=embeddings.device)
        n_valid = 0
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

        return loss, per_label_loss


# ---------------------------------------------------------------------------
# Model — reuses SiameseNetwork architecture from CL_COREF
# ---------------------------------------------------------------------------

class CorefEncoder(nn.Module):
    """
    Coreference mention encoder: RoBERTa-large + dual MLP heads.

    Produces a 2048-dim embedding per mention by concatenating:
      - CLS token embedding projected through MLP (1024-dim)
      - Mention span embedding (sum of [E]...[/E] tokens) projected through MLP (1024-dim)

    Same architecture as CL_COREF/model.py SiameseNetwork, adapted for
    standalone use with MS Loss (no paired forward pass needed).
    """

    def __init__(self, tokenizer, mlp_hidden_dim=1024):
        super().__init__()
        self.language_model = RobertaModel.from_pretrained("roberta-large")
        self.language_model.resize_token_embeddings(len(tokenizer))

        self.e_start_id = tokenizer.convert_tokens_to_ids("[E]")
        self.e_end_id = tokenizer.convert_tokens_to_ids("[/E]")
        self.mlp_hidden_dim = mlp_hidden_dim
        hidden_size = self.language_model.config.hidden_size  # 1024

        self.mlp_cls = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )
        self.mlp_mention = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, L) token ids.
            attention_mask: (B, L) attention mask.

        Returns:
            (B, 2048) L2-normalized embeddings.
        """
        # Find [E] and [/E] positions
        e_start_idx = input_ids.eq(self.e_start_id).nonzero(as_tuple=True)
        e_end_idx = input_ids.eq(self.e_end_id).nonzero(as_tuple=True)

        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        # CLS embedding
        cls_embed = hidden[:, 0, :]  # (B, H)

        # Mention span embedding: sum tokens between [E] and [/E]
        seq_indices = torch.arange(hidden.size(1), device=hidden.device)
        start_mask = seq_indices >= e_start_idx[1].unsqueeze(1)
        end_mask = seq_indices <= e_end_idx[1].unsqueeze(1)
        mention_mask = start_mask & end_mask  # (B, L)
        mention_embed = (hidden * mention_mask.unsqueeze(2).float()).sum(dim=1)  # (B, H)

        # Project through MLPs and concatenate
        out_cls = self.mlp_cls(cls_embed)  # (B, mlp_dim)
        out_mention = self.mlp_mention(mention_embed)  # (B, mlp_dim)
        concat = torch.cat([out_cls, out_mention], dim=1)  # (B, 2*mlp_dim)

        # L2 normalize for cosine-based MS Loss
        return F.normalize(concat, p=2, dim=1)


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
# Validation (embedding + clustering + pd_LEA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """
    Compute embeddings for all mentions in a dataloader.

    Returns:
        embeddings: (N, D) numpy array of L2-normalized embeddings.
        labels: list of string chain IDs.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        embeddings = model(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(batch["label"])

    return np.concatenate(all_embeddings, axis=0), all_labels


def _compute_lea(input_clusters, mention_to_output, weight_fn):
    """
    Generic LEA computation with pluggable cluster weighting.

    Args:
        input_clusters: list of sets (each set = mention indices in one cluster).
        mention_to_output: dict mapping mention index -> set of mentions in its
            output cluster.
        weight_fn: callable(cluster) -> float, the importance weight for a cluster.

    Returns:
        (numerator, denominator) for recall or precision.
    """
    num, den = 0.0, 0.0
    for cluster in input_clusters:
        w = weight_fn(cluster)
        if len(cluster) == 1:
            m = next(iter(cluster))
            den += w
            if len(mention_to_output.get(m, set())) == 1:
                num += w
        else:
            den += w
            all_links = len(cluster) * (len(cluster) - 1) / 2.0
            common_links = 0.0
            for m in cluster:
                out_cluster = mention_to_output.get(m, set())
                common_links += len(cluster & out_cluster) - 1
            common_links /= 2.0
            num += w * (common_links / all_links) if all_links > 0 else 0.0
    return num, den


def evaluate_clustering(embeddings, gold_labels, pd_tuples, threshold=0.5):
    """
    Agglomerative clustering + LEA and pd_LEA evaluation.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        gold_labels: list of string chain IDs (len N).
        pd_tuples: list of (mention_head, mention_wordset, mention_wo_stopwords)
            tuples (len N). Used for pd_LEA weighting.
        threshold: cosine distance threshold for clustering.

    Returns:
        dict with lea_* and pd_lea_* metrics.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    distances = pdist(embeddings, metric="cosine")
    Z = linkage(distances, method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")

    # Build cluster -> mention-indices mapping
    pred_clusters = defaultdict(set)
    gold_clusters = defaultdict(set)
    for i, (pred_id, gold_id) in enumerate(zip(cluster_ids, gold_labels)):
        pred_clusters[pred_id].add(i)
        gold_clusters[gold_id].add(i)

    pred_list = list(pred_clusters.values())
    gold_list = list(gold_clusters.values())

    mention_to_gold = {}
    for mentions in gold_clusters.values():
        for m in mentions:
            mention_to_gold[m] = mentions

    mention_to_pred = {}
    for mentions in pred_clusters.values():
        for m in mentions:
            mention_to_pred[m] = mentions

    # --- Standard LEA (weight = cluster size) ---
    def lea_weight(cluster):
        return len(cluster)

    lea_r_num, lea_r_den = _compute_lea(gold_list, mention_to_pred, lea_weight)
    lea_p_num, lea_p_den = _compute_lea(pred_list, mention_to_gold, lea_weight)

    lea_r = lea_r_num / lea_r_den if lea_r_den > 0 else 0
    lea_p = lea_p_num / lea_p_den if lea_p_den > 0 else 0
    lea_f1 = 2 * lea_p * lea_r / (lea_p + lea_r) if (lea_p + lea_r) > 0 else 0

    # --- pd_LEA (weight = phrasing diversity of cluster) ---
    def pd_weight(cluster):
        if len(cluster) == 1:
            return 0.0
        mentions = [pd_tuples[i] for i in cluster if pd_tuples[i][2] is not None]
        if len(mentions) < 2:
            return 0.0
        return phrasing_diversity_calc(mentions)

    pd_r_num, pd_r_den = _compute_lea(gold_list, mention_to_pred, pd_weight)
    pd_p_num, pd_p_den = _compute_lea(pred_list, mention_to_gold, pd_weight)

    pd_lea_r = pd_r_num / pd_r_den if pd_r_den > 0 else 0
    pd_lea_p = pd_p_num / pd_p_den if pd_p_den > 0 else 0
    pd_lea_f1 = (
        2 * pd_lea_p * pd_lea_r / (pd_lea_p + pd_lea_r)
        if (pd_lea_p + pd_lea_r) > 0
        else 0
    )

    return {
        "lea_precision": lea_p,
        "lea_recall": lea_r,
        "lea_f1": lea_f1,
        "pd_lea_precision": pd_lea_p,
        "pd_lea_recall": pd_lea_r,
        "pd_lea_f1": pd_lea_f1,
        "n_gold_clusters": len(gold_list),
        "n_pred_clusters": len(pred_list),
    }


def build_val_resources(mention_df, tokenizer, max_length, val_batch_size, num_workers):
    """
    Build per-dataset validation dataloaders and PD tuples.

    Returns:
        val_resources: dict of dataset_name -> {
            "dataloader": DataLoader,
            "pd_tuples": list of (head, wordset, wo_stopwords) per mention,
        }
    """
    val_resources = {}
    val_splits = mention_df[
        mention_df["split"].isin(["val", "validation"])
    ]

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
            # Keep None for mentions without PD components
            if isinstance(wordset, str) and isinstance(wo_stop, str):
                pd_tuples.append((head, wordset, wo_stop))
            else:
                pd_tuples.append((head, None, None))

        val_resources[dataset_name] = {
            "dataloader": dataloader,
            "pd_tuples": pd_tuples,
            "n_mentions": len(ds_df),
        }

    return val_resources


def run_all_validations(model, val_resources, device, threshold):
    """
    Run validation on all datasets, return per-dataset metrics.

    Args:
        model: CorefEncoder (unwrapped from DDP).
        val_resources: output of build_val_resources.
        device: torch device.
        threshold: clustering distance threshold.

    Returns:
        dict of dataset_name -> metrics dict.
    """
    all_metrics = {}
    for dataset_name, res in val_resources.items():
        embeddings, labels = compute_embeddings(model, res["dataloader"], device)
        metrics = evaluate_clustering(
            embeddings, labels, res["pd_tuples"], threshold=threshold
        )
        all_metrics[dataset_name] = metrics
    return all_metrics


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

    # --- Tokenizer ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])

    # --- Build training dataloader ---
    per_gpu_batch = args.batch_size // args.world_size
    train_dataloader, chain_index = build_pk_dataloader(
        mention_df=mention_df,
        tokenizer=tokenizer,
        batch_size=per_gpu_batch if is_distributed else args.batch_size,
        k=args.k,
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
    val_resources = build_val_resources(
        mention_df, tokenizer, args.max_length, args.val_batch_size, args.num_workers
    )
    if is_main:
        for ds_name, res in sorted(val_resources.items()):
            logger.info(f"  Val {ds_name:20s}: {res['n_mentions']:>6,} mentions")
        logger.info(f"  Monitor metric: avg pd_LEA F1 across all {len(val_resources)} datasets")

    # --- Model ---
    model = CorefEncoder(tokenizer, mlp_hidden_dim=args.mlp_dim)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

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

    # --- Checkpointing ---
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training ---
    if is_main:
        logger.info(
            f"Starting training: {args.num_epochs} epochs, "
            f"batch_size={args.batch_size}, micro_batch={args.micro_batch_size}, "
            f"lr={args.lr}"
        )

    best_val_f1 = 0.0
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_chain_losses = defaultdict(list)
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
            emb_chunks = []
            with torch.no_grad():
                for start in range(0, B, mb):
                    end = min(start + mb, B)
                    with torch.amp.autocast("cuda", enabled=args.fp16):
                        chunk_emb = model(
                            input_ids[start:end], attention_mask[start:end]
                        )
                    emb_chunks.append(chunk_emb)

            all_emb = torch.cat(emb_chunks, dim=0)  # (B, 2048) detached

            # Cross-GPU gathering (on detached embeddings)
            if is_distributed:
                all_emb_gathered, all_labels = gather_embeddings(
                    all_emb, int_labels
                )
            else:
                all_emb_gathered, all_labels = all_emb, int_labels

            # --- Phase 2: Compute loss and get embedding gradients ---
            all_emb_gathered.requires_grad_(True)
            with torch.amp.autocast("cuda", enabled=args.fp16):
                loss, per_label_losses = ms_loss_fn(all_emb_gathered, all_labels)

            # Backprop through the loss to get gradients on embeddings only
            # (no model graph involved yet — this is cheap)
            scaler.scale(loss).backward()
            # emb_grad shape: (B_gathered, 2048) — gradient of loss w.r.t. embeddings
            emb_grad = all_emb_gathered.grad.detach().clone()

            # If distributed, slice back to local shard
            if is_distributed:
                rank = dist.get_rank()
                emb_grad = emb_grad[rank * B : (rank + 1) * B]

            # --- Phase 3: Re-forward each micro-batch WITH grad ---
            # Use pre-computed emb_grad to drive backward through model.
            # Only one micro-batch's activations are in memory at a time.
            optimizer.zero_grad()
            for start in range(0, B, mb):
                end = min(start + mb, B)
                with torch.amp.autocast("cuda", enabled=args.fp16):
                    chunk_emb = model(
                        input_ids[start:end], attention_mask[start:end]
                    )
                # Backprop this chunk using pre-computed embedding gradients.
                # This is equivalent to: (d_loss/d_emb) * (d_emb/d_theta)
                chunk_emb.backward(emb_grad[start:end])

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track losses
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            # Accumulate per-chain losses for difficulty updates
            for label_int, losses in per_label_losses.items():
                # Map back to string label
                for str_label, int_label in label_encoder._map.items():
                    if int_label == label_int:
                        epoch_chain_losses[str_label].extend(losses)
                        break

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
            logger.info(
                f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, "
                f"time={epoch_time:.0f}s"
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

        # --- Validation ---
        if is_main and (epoch + 1) % args.val_every == 0:
            logger.info("Running validation on all datasets...")
            raw_model = model.module if is_distributed else model
            all_metrics = run_all_validations(
                raw_model, val_resources, device, args.cluster_threshold
            )

            # Log per-dataset metrics
            for ds_name in sorted(all_metrics.keys()):
                m = all_metrics[ds_name]
                logger.info(
                    f"  {ds_name:20s}  "
                    f"LEA F1={m['lea_f1']:.4f} (P={m['lea_precision']:.4f} R={m['lea_recall']:.4f})  "
                    f"pd_LEA F1={m['pd_lea_f1']:.4f} (P={m['pd_lea_precision']:.4f} R={m['pd_lea_recall']:.4f})  "
                    f"clusters: gold={m['n_gold_clusters']} pred={m['n_pred_clusters']}"
                )

            # Monitor metric: mean pd_LEA F1 across all datasets
            avg_pd_lea_f1 = np.mean([m["pd_lea_f1"] for m in all_metrics.values()])
            logger.info(f"  Avg pd_LEA F1 (all datasets) = {avg_pd_lea_f1:.4f}")

            # Save best checkpoint
            if avg_pd_lea_f1 > best_val_f1:
                best_val_f1 = avg_pd_lea_f1
                ckpt_path = ckpt_dir / "best_model.pt"
                _safe_torch_save({
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "best_avg_pd_lea_f1": best_val_f1,
                    "all_metrics": {
                        ds: {k: float(v) for k, v in m.items()}
                        for ds, m in all_metrics.items()
                    },
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"  New best model saved: avg pd_LEA F1={best_val_f1:.4f}")

        # Save periodic checkpoint (model-only to stay within filesystem limits)
        if is_main and (epoch + 1) % args.save_every == 0:
            raw_model = model.module if is_distributed else model
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
            _safe_torch_save({
                "epoch": epoch + 1,
                "model_state_dict": raw_model.state_dict(),
                "best_pd_lea_f1": best_val_f1,
                "args": vars(args),
            }, ckpt_path)

    if is_main:
        logger.info(f"Training complete. Best avg pd_LEA F1: {best_val_f1:.4f}")

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MS Loss coreference training")

    # Data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str,
                   default="/Volumes/jnj_polyphonic_deid/default/cdcr/ms_coref")
    p.add_argument("--max-length", type=int, default=128)

    # PK Sampling
    p.add_argument("--batch-size", type=int, default=2048,
                    help="Total batch size across all GPUs")
    p.add_argument("--micro-batch-size", type=int, default=128,
                    help="Forward pass chunk size per GPU (controls GPU memory)")
    p.add_argument("--k", type=int, default=2)
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
    p.add_argument("--num-epochs", type=int, default=50)
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
    p.add_argument("--cluster-threshold", type=float, default=0.5)

    # Checkpointing
    p.add_argument("--save-every", type=int, default=5)

    # System
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--world-size", type=int, default=1,
                    help="Number of GPUs (auto-detected with torchrun)")

    # parse_known_args ignores unrecognized args (e.g. Databricks kernel's -f flag)
    args, _ = p.parse_known_args()

    # Auto-detect world size from torchrun / TorchDistributor
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
