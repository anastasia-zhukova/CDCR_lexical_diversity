"""
Ray Tune hyperparameter sweep for MS Loss coreference training.

Runs 8 parallel single-GPU trials on 8×H100 with ASHA early stopping.
Each trial is a full train() call (single-GPU mode) reporting pd_LEA F1,
LEA F1, and CoNLL F1 at each epoch; ASHA pruning, checkpoint selection,
and best-trial selection all use whichever of the three is configured via
config.yaml's `selection_metric` (default pd_lea_f1).

Usage on Databricks:
    1. Install: %pip install "ray[tune]"
    2. Run this script directly: %run ./ray_tune_sweep

Alternatively from terminal:
    python ray_tune_sweep.py --data-dir data --output-dir /Volumes/.../ms_coref/sweep
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import torch
import yaml

from ray import tune, train as ray_train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.air.config import CheckpointConfig

# Silence the slow experiment-state checkpoint warning on remote/DBFS storage.
# The warning fires when syncing the global experiment state takes longer than
# TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S (default 30s). On DBFS
# this is expected, so raise the threshold rather than spamming the log.
os.environ.setdefault("TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S", "600")

# On Databricks serverless there is no local disk — everything goes to DBFS,
# which is slow for frequent syncs. Disable the experiment-state checkpoint
# entirely; we don't need to restore a crashed sweep mid-run on serverless.
os.environ.setdefault("TUNE_DISABLE_EXPERIMENT_STATE_CHECKPOINT_SYNC", "1")

from core.coref_training_core import (
    MultiSimilarityLoss,
    ContrastiveLoss,
    LabelEncoder,
    load_and_filter_mentions,
    build_model_and_data,
    build_cosine_schedule,
    run_train_epoch,
    evaluate_validation,
)

logger = logging.getLogger(__name__)


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
# ---------------------------------------------------------------------------
_CFG = _load_config()

DATASETS = _CFG["datasets"]
EXCLUDE_TOPICS = _CFG["exclude_topics"]
HYPERCOREF_EXCLUDE_TOPICS = set(_CFG["hypercoref_exclude_topics"])
WECENG_EXCLUDE_TOPICS = set(_CFG["weceng_exclude_topics"])
TRAIN_MODE = _CFG["train_mode"]
SINGLE_TRAIN_DATASET = _CFG["single_train_dataset"]
EVAL_THRESHOLD = _CFG["eval_threshold"]
EVAL_LEVEL = _CFG.get("eval_level", "subtopic")

# Metric used for ASHA pruning, checkpoint selection, and best-trial
# selection. "pd_lea_f1" (default), "lea_f1", or "conll_f1" — all three are
# always computed and reported per epoch regardless of which one is used
# for selection (see evaluate_validation() in core/coref_training_core.py).
SELECTION_METRIC = _CFG.get("selection_metric", "pd_lea_f1")
if SELECTION_METRIC not in ("pd_lea_f1", "lea_f1", "conll_f1"):
    raise ValueError(
        f"config.yaml selection_metric must be one of 'pd_lea_f1', 'lea_f1', "
        f"'conll_f1', got {SELECTION_METRIC!r}"
    )

# Training regime — see core/coref_training_core.py. "ms_loss" (default) or
# "contrastive". Determines both the trial's model/data wiring below and
# which search space run_sweep() uses.
LOSS_TYPE = _CFG.get("loss_type", "ms_loss")


def train_trial(config, data_dir="data"):
    """
    Single-GPU training function called by Ray Tune.

    Args:
        config: dict of hyperparameters sampled by Tune.
        data_dir: path to data directory (passed via tune.with_parameters or partial).
    """
    import traceback, sys

    try:
        _train_trial_inner(config, data_dir)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise


def _train_trial_inner(config, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    mention_df, train_mention_df, train_datasets = load_and_filter_mentions(
        data_dir,
        datasets=DATASETS,
        train_mode=TRAIN_MODE,
        single_train_dataset=SINGLE_TRAIN_DATASET,
        exclude_topics=EXCLUDE_TOPICS,
        hypercoref_exclude_topics=HYPERCOREF_EXCLUDE_TOPICS,
        weceng_exclude_topics=WECENG_EXCLUDE_TOPICS,
    )

    # --- Tokenizer, dataloaders, and model ---
    val_datasets = [SINGLE_TRAIN_DATASET] if TRAIN_MODE == "single" else None
    tokenizer, train_dataloader, chain_index, val_resources, model = (
        build_model_and_data(
            mention_df,
            train_mention_df,
            training_mode=LOSS_TYPE,
            batch_size=config.get("batch_size", 2048),
            max_length=128,
            tau=config.get("tau", 1.0),
            gamma=config.get("gamma", 0.5),
            singleton_frac=config.get("singleton_frac", 0.20),
            num_workers=0,
            seed=42,
            val_batch_size=256,
            mlp_dim=config.get("mlp_dim", 1024),
            device=device,
            val_datasets=val_datasets,
            contrastive_sbert_model=config.get("contrastive_sbert_model"),
            contrastive_k_negatives=config.get("contrastive_k_negatives"),
            eval_level=EVAL_LEVEL,
        )
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    num_epochs = config.get("num_epochs", 25)
    total_steps = num_epochs * len(train_dataloader)
    scheduler = build_cosine_schedule(
        optimizer, total_steps, config.get("warmup_frac", 0.05)
    )

    # --- Loss ---
    if LOSS_TYPE == "ms_loss":
        loss_fn = MultiSimilarityLoss(
            alpha=config.get("ms_alpha", 2.0),
            beta=config["ms_beta"],
            lam=config["ms_lambda"],
        )
        label_encoder = LabelEncoder()
    else:  # contrastive
        loss_fn = ContrastiveLoss(margin=config.get("contrastive_margin", 0.7))
        label_encoder = None

    # --- Training loop ---
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    mb = config.get("micro_batch_size", 128)
    grad_clip = config.get("grad_clip", 1.0)

    best_score = float("-inf")

    for epoch in range(num_epochs):
        if LOSS_TYPE == "ms_loss":
            epoch_stats = run_train_epoch(
                LOSS_TYPE,
                model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_fn,
                label_encoder,
                scaler,
                device,
                micro_batch_size=mb,
                grad_clip=grad_clip,
                fp16=True,
                is_distributed=False,
                track_chain_losses=False,  # skip per-chain loss tracking for sweep speed
                loss_in_fp32=True,
            )
        else:  # contrastive
            epoch_stats = run_train_epoch(
                LOSS_TYPE,
                model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_fn,
                scaler,
                device,
                micro_batch_size=mb,
                grad_clip=grad_clip,
                fp16=True,
            )
        avg_loss = epoch_stats["avg_loss"]

        # --- Validation ---
        eval_metrics, avg_metrics = evaluate_validation(
            model,
            val_resources,
            device,
            threshold=EVAL_THRESHOLD,
        )

        metrics = {
            "epoch": epoch + 1,
            "pd_lea_f1": float(avg_metrics["pd_lea_f1"]),
            "conll_f1": float(avg_metrics["conll_f1"]),
            "lea_f1": float(avg_metrics["lea_f1"]),
            "train_loss": avg_loss,
        }
        sel_value = avg_metrics[SELECTION_METRIC]

        # Only write a checkpoint when validation score improves. Databricks
        # serverless has no local disk (everything goes to slow DBFS/Volumes),
        # so we avoid writing on every epoch and only persist "promising"
        # checkpoints. CheckpointConfig(checkpoint_score_attribute=SELECTION_METRIC)
        # then keeps just the best one on disk, deleting superseded ones.
        if sel_value > best_score:
            best_score = sel_value
            with tempfile.TemporaryDirectory() as tmp_dir:
                torch.save(model.state_dict(), os.path.join(tmp_dir, "model.pt"))
                checkpoint = ray_train.Checkpoint.from_directory(tmp_dir)
                # Report to Ray Tune (ASHA uses this to prune) with checkpoint
                ray_train.report(metrics, checkpoint=checkpoint)
        else:
            # Report to Ray Tune (ASHA uses this to prune) without checkpoint
            ray_train.report(metrics)


def run_sweep(args):
    """Configure and launch the Ray Tune sweep."""

    if LOSS_TYPE == "ms_loss":
        search_space = {
            "lr": tune.loguniform(5e-6, 8e-5),
            "ms_beta": tune.uniform(20.0, 80.0),
            "ms_lambda": tune.uniform(0.3, 0.7),
            "tau": tune.uniform(0.5, 2.0),
            "ms_alpha": tune.uniform(1.0, 4.0),
            "warmup_frac": tune.uniform(0.03, 0.10),
            "weight_decay": tune.loguniform(1e-3, 0.1),
            # Fixed for all trials:
            "batch_size": 2048,
            "micro_batch_size": 128,
            "num_epochs": args.num_epochs,
            "mlp_dim": 1024,
            "gamma": 0.5,
            "singleton_frac": 0.20,
            "grad_clip": 1.0,
        }
    else:  # contrastive — no PK sampler / MS-loss-specific knobs to tune
        search_space = {
            "lr": tune.loguniform(5e-6, 8e-5),
            "contrastive_margin": tune.uniform(0.3, 0.9),
            "warmup_frac": tune.uniform(0.03, 0.10),
            "weight_decay": tune.loguniform(1e-3, 0.1),
            # Fixed for all trials:
            "batch_size": 256,
            "num_epochs": args.num_epochs,
            "mlp_dim": 1024,
            "grad_clip": 1.0,
        }

    scheduler = ASHAScheduler(
        metric=SELECTION_METRIC,
        mode="max",
        max_t=args.num_epochs,
        grace_period=5,
        reduction_factor=3,
    )

    reporter = CLIReporter(
        metric_columns=["epoch", "pd_lea_f1", "lea_f1", "conll_f1", "train_loss"],
        max_report_frequency=60,
    )

    trainable = tune.with_parameters(train_trial, data_dir=args.data_dir)
    trainable = tune.with_resources(trainable, {"gpu": 1})

    # Keep only the single best-scoring checkpoint per trial (train_trial only
    # ever reports a checkpoint when SELECTION_METRIC improves, so writes to
    # the slow DBFS/Volumes storage path are already rare).
    # checkpoint_score_attribute ensures Ray keeps the highest-scoring
    # checkpoint rather than just the most recent one, deleting superseded
    # checkpoints automatically.
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute=SELECTION_METRIC,
        checkpoint_score_order="max",
    )

    # On serverless (no local disk) we point storage_path at DBFS, which is
    # slow for syncs. Use a no-op syncer so Ray does NOT copy per-trial
    # checkpoint directories back to the driver/storage during the run.
    # Results are still available in-memory via the ResultGrid returned by
    # tuner.fit(); we just skip the expensive remote sync of large tensors.
    sync_config = tune.SyncConfig()

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_trials,
        ),
        run_config=tune.RunConfig(
            name=f"ms_coref_sweep_{LOSS_TYPE}",
            storage_path=args.output_dir,
            progress_reporter=reporter,
            checkpoint_config=checkpoint_config,
            sync_config=sync_config,
            # Reduce experiment-state sync frequency. The default syncs very
            # often; on DBFS each sync can take many minutes with 50 trials.
            verbose=1,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(SELECTION_METRIC, mode="max")
    print("\n" + "=" * 60)
    print(f"BEST TRIAL (selected on {SELECTION_METRIC})")
    print("=" * 60)
    print(f"  pd_LEA F1: {best_result.metrics['pd_lea_f1']:.4f}")
    print(f"  LEA F1:    {best_result.metrics['lea_f1']:.4f}")
    print(f"  CoNLL F1:  {best_result.metrics['conll_f1']:.4f}")
    print(f"  Config:    {best_result.config}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ray Tune HPO sweep for MS Loss CDCR")
    p.add_argument("--data-dir", type=str, default=_CFG.get("data_dir"))
    p.add_argument("--output-dir", type=str, default=_CFG.get("output_dir"))
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--num-epochs", type=int, default=25)
    args, _ = p.parse_known_args()

    run_sweep(args)
