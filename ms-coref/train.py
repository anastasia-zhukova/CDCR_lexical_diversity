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
import json
import logging
import os
import subprocess
import sys
import tempfile
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.coref_training_core import (
    MultiSimilarityLoss,
    LabelEncoder,
    load_and_filter_mentions,
    build_model_and_data,
    build_cosine_schedule,
    run_train_epoch,
    evaluate_validation,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Databricks/Spark infrastructure noise
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _load_config():
    base = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Experiment configuration — read from config.yaml here (not in
# coref_training_core.py, which stays independent of any one script's
# configuration) and passed explicitly into the shared helper functions.
# MultiSimilarityLoss, LabelEncoder, build_val_resources(), and
# build_model_and_data() live in coref_training_core.py, shared with
# ray_tune_sweep.py.
# ---------------------------------------------------------------------------
_CFG = _load_config()

DATASETS = _CFG.get(
    "datasets", ["HyperCoref", "NP4E", "NewsWCL50", "NIdent", "ECBplus", "ECBplusMETAm"]
)
EXCLUDE_TOPICS = _CFG.get("exclude_topics", True)
HYPERCOREF_EXCLUDE_TOPICS = set(_CFG.get("hypercoref_exclude_topics", ["Sport", "Sports"]))
WECENG_EXCLUDE_TOPICS = set(_CFG.get("weceng_exclude_topics", [
    "Awards", "Beauty Pageant", "Concert", "Contest", "Festival", "Solar Eclipse",
]))
TRAIN_MODE = _CFG.get("train_mode", "mix")
SINGLE_TRAIN_DATASET = _CFG.get("single_train_dataset", "HyperCoref")
EVAL_THRESHOLD = _CFG.get("eval_threshold", 0.5)



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
    mention_df, train_mention_df, train_datasets = load_and_filter_mentions(
        args.data_dir,
        datasets=DATASETS,
        train_mode=TRAIN_MODE,
        single_train_dataset=SINGLE_TRAIN_DATASET,
        exclude_topics=EXCLUDE_TOPICS,
        hypercoref_exclude_topics=HYPERCOREF_EXCLUDE_TOPICS,
        weceng_exclude_topics=WECENG_EXCLUDE_TOPICS,
        log=logger.info if is_main else None,
    )

    # --- Tokenizer, dataloaders, and model ---
    # In single mode, evaluate only on the training dataset's own val split.
    val_datasets = [SINGLE_TRAIN_DATASET] if TRAIN_MODE == "single" else None
    per_gpu_batch = args.batch_size // args.world_size
    wrap_model_fn = (
        (lambda m: DDP(m, device_ids=[local_rank], find_unused_parameters=False))
        if is_distributed else None
    )
    tokenizer, train_dataloader, chain_index, val_resources, model = build_model_and_data(
        mention_df,
        train_mention_df,
        batch_size=per_gpu_batch if is_distributed else args.batch_size,
        max_length=args.max_length,
        tau=args.tau,
        gamma=args.gamma,
        singleton_frac=args.singleton_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        val_batch_size=args.val_batch_size,
        mlp_dim=args.mlp_dim,
        device=device,
        val_datasets=val_datasets,
        wrap_model_fn=wrap_model_fn,
        log=logger.info if is_main else None,
    )
    if is_main:
        if TRAIN_MODE == "single":
            logger.info(f"  Monitor metric: pd_LEA F1 on {SINGLE_TRAIN_DATASET} val")
        else:
            logger.info(f"  Monitor metric: avg pd_LEA F1 across all {len(val_resources)} datasets")
        logger.info(f"  Clustering: per-subtopic (matching CDCR evaluation protocol)")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup + cosine decay
    total_steps = args.num_epochs * len(train_dataloader)
    scheduler = build_cosine_schedule(optimizer, total_steps, args.warmup_frac)

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
        local_log_path = Path(tempfile.mktemp(
            prefix=f"ms_coref_{args.run_suffix}_", suffix="_train.log", dir="/tmp"))
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
        t0 = time.time()

        def _log_step(batch_idx, num_batches, loss_val):
            nonlocal global_step
            global_step += 1
            if is_main and global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}  "
                    f"Step {batch_idx+1}/{num_batches}  "
                    f"Loss: {loss_val:.4f}  "
                    f"LR: {lr:.2e}"
                )

        epoch_stats = run_train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            ms_loss_fn,
            label_encoder,
            scaler,
            device,
            micro_batch_size=args.micro_batch_size,
            grad_clip=args.grad_clip,
            fp16=args.fp16,
            is_distributed=is_distributed,
            track_chain_losses=True,
            step_callback=_log_step,
        )
        avg_loss = epoch_stats["avg_loss"]
        epoch_steps = epoch_stats["epoch_steps"]

        # --- End of epoch ---
        epoch_time = time.time() - t0

        if is_main:
            avg_pos = epoch_stats["mined_pos"] / max(epoch_steps, 1)
            avg_neg = epoch_stats["mined_neg"] / max(epoch_steps, 1)
            avg_anchors = epoch_stats["valid_anchors"] / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, "
                f"time={epoch_time:.0f}s  |  "
                f"avg mined pairs/batch: pos={avg_pos:.1f}  neg={avg_neg:.1f}  "
                f"valid_anchors={avg_anchors:.1f}"
            )

        # Update chain difficulty EMA
        chain_loss_means = epoch_stats["chain_loss_means"]
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
            eval_metrics, avg_pd_lea_f1, avg_conll_f1 = evaluate_validation(
                raw_model, val_resources, device, threshold=EVAL_THRESHOLD,
            )
            dataset_names = sorted(val_resources.keys())

            if device.type == "cuda":
                torch.cuda.empty_cache()

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
            logger.info(f"\nRunning eval_checkpoint.py on test set with {ckpt_for_eval.name} ...")
            eval_script = Path(os.getcwd()) / "eval_checkpoint.py"
            cmd = [
                sys.executable, str(eval_script),
                "--checkpoint", str(ckpt_for_eval),
                "--data-dir", args.data_dir,
                "--output-dir", str(run_dir / "conll_output"),
                "--split", "test",
                "--setting", args.run_suffix,
                "--max-length", str(args.max_length),
                "--mlp-dim", str(args.mlp_dim),
                "--batch-size", str(args.val_batch_size),
                "--exclude-topics" if EXCLUDE_TOPICS else "--no-exclude-topics",
            ]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                logger.warning(f"eval_checkpoint.py exited with code {result.returncode}")
            else:
                logger.info("eval_checkpoint.py completed successfully.")
        else:
            logger.warning(f"No checkpoint found at {ckpt_for_eval}, skipping eval_checkpoint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    cfg = _CFG
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
