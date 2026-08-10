"""
Coreference training for cross-document coreference resolution.

Supports two pluggable training regimes (see coref_training_core.py):
  - "ms_loss":      MultiSimilarityLoss + PK batch sampling with structured
                    negative difficulty and cross-GPU embedding gathering.
  - "contrastive":  Margin-based ContrastiveLoss + static SBERT MST/KNN pair
                    sampling (ported from cl-coref-orig).

Both regimes share the same encoder architecture and pd_LEA validation
protocol across 6 CDCR datasets.

Training modes (configured via constants at top of file):
  - single:          Train on one dataset, evaluate on all
  - hypercoref_main: Train on HyperCoref, evaluate on all
  - mix:             Train on all datasets combined
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import yaml
import time
from pathlib import Path

import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from transformers import RobertaTokenizer

from coref_encoder import compute_embeddings, evaluate_per_subtopic
from coref_training_core import (
    build_model_and_data,
    load_and_filter_mentions,
    run_train_epoch,
)
from pair_sampler import K_HARD_NEGATIVES, SBERT_MODEL
from sampler import MentionDataset
from torch.utils.data import DataLoader

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

# HyperCoref topic exclusion: Sport/Sports topics are dominated by BBC Sport content
# (54 subtopics, ~32K mentions, ~53% of HyperCoref). Excluded because sport entities
# and events cluster too easily by lexical overlap (player/team names), potentially
# inflating metrics and masking generalisation ability.
# Set to None or empty set to disable.
HYPERCOREF_EXCLUDE_TOPICS = {"Sport", "Sports"}

# Training mode: "single", "hypercoref_main", "mix"
#   single          – train on one dataset at a time (loop over DATASETS externally)
#   hypercoref_main – train on HyperCoref, evaluate on all DATASETS
#   mix             – train on all DATASETS combined
TRAIN_MODE = "mix"

# When TRAIN_MODE == "single", which dataset to train on.
# Ignored for other modes.  Change this (or loop externally) per run.
SINGLE_TRAIN_DATASET = "HyperCoref"

# Fixed clustering threshold for cosine-distance evaluation
EVAL_THRESHOLD = 0.5


def build_val_resources(mention_df, tokenizer, max_length, val_batch_size, num_workers):
    """
    Build per-dataset validation dataloaders, PD tuples, and subtopic IDs.

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
    val_splits = mention_df[mention_df["split"].isin(["val", "validation"])]

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

    # --- Load data (shared across training regimes) ---
    mention_df, train_mention_df, train_datasets = load_and_filter_mentions(
        data_dir=args.data_dir,
        datasets=DATASETS,
        train_mode=TRAIN_MODE,
        single_train_dataset=SINGLE_TRAIN_DATASET,
        hypercoref_exclude_topics=HYPERCOREF_EXCLUDE_TOPICS,
        is_main=is_main,
    )

    # --- Tokenizer ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])

    # --- Build model, training dataloader, and loss for the selected regime ---
    model, train_dataloader, loss_fn, extra = build_model_and_data(
        training_mode=args.loss_type,
        train_mention_df=train_mention_df,
        tokenizer=tokenizer,
        args=args,
        is_distributed=is_distributed,
        is_main=is_main,
    )
    chain_index = extra["chain_index"]
    label_encoder = extra["label_encoder"]
    if is_main:
        logger.info(f"Training dataloader: {len(train_dataloader)} batches/epoch")

    # --- Build per-dataset validation dataloaders (always all DATASETS) ---
    val_resources = build_val_resources(
        mention_df, tokenizer, args.max_length, args.val_batch_size, args.num_workers
    )
    if is_main:
        for ds_name, res in sorted(val_resources.items()):
            logger.info(
                f"  Val {ds_name:20s}: {res['n_mentions']:>6,} mentions  "
                f"{res['n_subtopics']:>4} subtopics"
            )
        logger.info(
            f"  Monitor metric: avg pd_LEA F1 across all {len(val_resources)} datasets"
        )
        logger.info(f"  Clustering: per-subtopic (matching CDCR evaluation protocol)")

    # --- Model ---
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {n_params / 1e6:.1f}M params ({n_trainable / 1e6:.1f}M trainable)"
        )

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Linear warmup + cosine decay
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
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
        # Two checkpoints, both selected on the training dataset's own val split
        best_pd_lea = 0.0
        best_conll = 0.0
    else:
        # One checkpoint selected by avg pd_LEA F1 across all datasets
        best_val_f1 = 0.0

    # Per-epoch metrics accumulated in memory, written fully each epoch
    # (avoids Illegal seek on Databricks volume filesystems which don't support append mode)
    metrics_log_path = run_dir / "metrics_log.jsonl"
    metrics_log_rows = []

    # --- Training ---
    if is_main:
        logger.info(
            f"Starting training: {args.num_epochs} epochs, "
            f"batch_size={args.batch_size}, lr={args.lr}"
        )

    global_step = 0

    for epoch in range(args.num_epochs):
        # This is the piece that changes across training regimes — see
        # coref_training_core.run_train_epoch (ms_loss vs contrastive).
        stats, global_step = run_train_epoch(
            training_mode=args.loss_type,
            model=model,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
            label_encoder=label_encoder,
            is_distributed=is_distributed,
            is_main=is_main,
        )

        # Update chain difficulty EMA (ms_loss regime only)
        if chain_index is not None:
            chain_index.update_difficulty(
                stats["chain_loss_means"], alpha=args.difficulty_alpha
            )
            if is_main and len(stats["chain_loss_means"]) > 0:
                vals = list(stats["chain_loss_means"].values())
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
                    embs,
                    labels,
                    res["pd_tuples"],
                    res["subtopic_ids"],
                    threshold=EVAL_THRESHOLD,
                )
                eval_metrics[ds_name] = metrics

            if device.type == "cuda":
                torch.cuda.empty_cache()

            avg_pd_lea_f1 = np.mean(
                [eval_metrics[ds]["pd_lea_f1"] for ds in dataset_names]
            )
            avg_conll_f1 = np.mean(
                [eval_metrics[ds]["conll_f1"] for ds in dataset_names]
            )

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
            logger.info(
                f"  Avg pd_LEA F1 = {avg_pd_lea_f1:.4f}  Avg CoNLL F1 = {avg_conll_f1:.4f}  (threshold={EVAL_THRESHOLD})"
            )

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
                sel_conll = sel_m["conll_f1"]

                if sel_pd_lea > best_pd_lea:
                    best_pd_lea = sel_pd_lea
                    _safe_torch_save(
                        {
                            **ckpt_payload,
                            "selection_metric": "pd_lea_f1",
                            "selection_dataset": SINGLE_TRAIN_DATASET,
                            "selection_value": sel_pd_lea,
                        },
                        ckpt_dir / "best_model_pd_lea.pt",
                    )
                    logger.info(
                        f"  New best (pd_LEA): {sel_pd_lea:.4f} on {SINGLE_TRAIN_DATASET} val"
                    )

                if sel_conll > best_conll:
                    best_conll = sel_conll
                    _safe_torch_save(
                        {
                            **ckpt_payload,
                            "selection_metric": "conll_f1",
                            "selection_dataset": SINGLE_TRAIN_DATASET,
                            "selection_value": sel_conll,
                        },
                        ckpt_dir / "best_model_conll.pt",
                    )
                    logger.info(
                        f"  New best (CoNLL): {sel_conll:.4f} on {SINGLE_TRAIN_DATASET} val"
                    )
            else:
                # mix / hypercoref_main: select by avg pd_LEA across all datasets
                if avg_pd_lea_f1 > best_val_f1:
                    best_val_f1 = avg_pd_lea_f1
                    _safe_torch_save(
                        {**ckpt_payload, "best_avg_pd_lea_f1": best_val_f1},
                        ckpt_dir / "best_model.pt",
                    )
                    logger.info(f"  New best (avg pd_LEA): {best_val_f1:.4f}")

            # Write metrics log — full overwrite each epoch (volume fs doesn't support append)
            metrics_log_rows.append(
                {
                    "epoch": epoch + 1,
                    "avg_pd_lea_f1": float(avg_pd_lea_f1),
                    "avg_conll_f1": float(avg_conll_f1),
                    "per_dataset": {
                        ds: {k: float(v) for k, v in m.items()}
                        for ds, m in eval_metrics.items()
                    },
                }
            )
            with open(metrics_log_path, "w") as f:
                for row in metrics_log_rows:
                    f.write(json.dumps(row) + "\n")

        # Save periodic checkpoint (model-only to stay within filesystem limits)
        if is_main and (epoch + 1) % args.save_every == 0:
            raw_model = model.module if is_distributed else model
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
            _safe_torch_save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "train_mode": TRAIN_MODE,
                    "train_datasets": train_datasets,
                    "args": vars(args),
                },
                ckpt_path,
            )

    if is_main:
        if TRAIN_MODE == "single":
            logger.info(
                f"Training complete. Best pd_LEA={best_pd_lea:.4f}  CoNLL={best_conll:.4f}  "
                f"(on {SINGLE_TRAIN_DATASET} val)"
            )
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
            logger.warning(
                f"Could not copy train log to volume: {e} (kept at {local_log_path})"
            )

    # --- Post-training inference on test set ---
    # Only run on main process, after distributed teardown.
    # Uses best_model_pd_lea.pt (single mode) or best_model.pt (mix/hypercoref_main).
    if is_main:
        if TRAIN_MODE == "single":
            ckpt_for_eval = ckpt_dir / "best_model_pd_lea.pt"
        else:
            ckpt_for_eval = ckpt_dir / "best_model.pt"

        if ckpt_for_eval.exists():
            logger.info(
                f"\nRunning conll_eval.py on test set with {ckpt_for_eval.name} ..."
            )
            conll_eval_script = Path(os.getcwd()) / "conll_eval.py"
            cmd = [
                sys.executable,
                str(conll_eval_script),
                "--checkpoint",
                str(ckpt_for_eval),
                "--data-dir",
                args.data_dir,
                "--output-dir",
                str(run_dir / "conll_output"),
                "--split",
                "test",
                "--setting",
                args.run_suffix,
                "--max-length",
                str(args.max_length),
                "--mlp-dim",
                str(args.mlp_dim),
                "--batch-size",
                str(args.val_batch_size),
            ]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                logger.warning(f"conll_eval.py exited with code {result.returncode}")
            else:
                logger.info("conll_eval.py completed successfully.")
        else:
            logger.warning(
                f"No checkpoint found at {ckpt_for_eval}, skipping conll_eval."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config():
    cfg_path = Path(__file__).parent / "config.yaml"
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
    p.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Subdirectory name for this run (default: auto from TRAIN_MODE/dataset)",
    )
    p.add_argument("--max-length", type=int, default=128)

    # PK Sampling
    p.add_argument(
        "--batch-size", type=int, default=2048, help="Total batch size across all GPUs"
    )
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=128,
        help="Forward pass chunk size per GPU (controls GPU memory)",
    )
    p.add_argument(
        "--tau", type=float, default=1.0, help="PD exponent for chain weighting"
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Difficulty EMA exponent for chain weighting",
    )
    p.add_argument("--singleton-frac", type=float, default=0.20)
    p.add_argument(
        "--difficulty-alpha",
        type=float,
        default=0.1,
        help="EMA smoothing for difficulty updates",
    )

    # Model
    p.add_argument("--mlp-dim", type=int, default=1024)

    # Training regime — see coref_training_core.py
    p.add_argument(
        "--loss-type",
        type=str,
        default="ms_loss",
        choices=["ms_loss", "contrastive"],
        help="ms_loss: MultiSimilarityLoss + PK batch sampler. "
        "contrastive: ContrastiveLoss + SBERT MST/KNN pair sampler.",
    )

    # MS Loss (loss-type=ms_loss)
    p.add_argument("--ms-alpha", type=float, default=2.0)
    p.add_argument("--ms-beta", type=float, default=50.0)
    p.add_argument("--ms-lambda", type=float, default=0.5)

    # Contrastive loss + pair sampler (loss-type=contrastive)
    p.add_argument(
        "--contrastive-margin",
        type=float,
        default=0.7,
        help="Margin for ContrastiveLoss (paper: 0.4 events, 0.7 entities)",
    )
    p.add_argument("--contrastive-sbert-model", type=str, default=SBERT_MODEL)
    p.add_argument(
        "--contrastive-k-negatives",
        type=int,
        default=K_HARD_NEGATIVES,
        help="Hard negatives per anchor mention in pair sampling",
    )

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
    p.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")

    # Checkpointing
    p.add_argument("--save-every", type=int, default=5)

    # System
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs (auto-detected with torchrun)",
    )

    args, _ = p.parse_known_args()

    # Auto-detect world size from torchrun / TorchDistributor
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Auto-derive run suffix from training config if not provided
    if args.run_suffix is None:
        all_six = [
            "HyperCoref",
            "NP4E",
            "NewsWCL50",
            "NIdent",
            "ECBplus",
            "ECBplusMETAm",
        ]
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

        if args.loss_type != "ms_loss":
            args.run_suffix = f"{args.run_suffix}_{args.loss_type}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
