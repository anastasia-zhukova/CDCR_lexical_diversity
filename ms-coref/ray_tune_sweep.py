"""
Ray Tune hyperparameter sweep for MS Loss coreference training.

Runs 8 parallel single-GPU trials on 8×H100 with ASHA early stopping.
Each trial is a full train() call (single-GPU mode) reporting pd_LEA F1
at each epoch for pruning.

Usage on Databricks:
    1. Install: %pip install "ray[tune]"
    2. Run this script directly: %run ./ray_tune_sweep

Alternatively from terminal:
    python ray_tune_sweep.py --data-dir data --output-dir /Volumes/.../ms_coref/sweep
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from ray import tune, train as ray_train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from core.coref_training_core import (
    MultiSimilarityLoss,
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
    tokenizer, train_dataloader, chain_index, val_resources, model = build_model_and_data(
        mention_df,
        train_mention_df,
        batch_size=config.get("batch_size", 2048),
        max_length=128,
        tau=config["tau"],
        gamma=config.get("gamma", 0.5),
        singleton_frac=config.get("singleton_frac", 0.20),
        num_workers=0,
        seed=42,
        val_batch_size=256,
        mlp_dim=config.get("mlp_dim", 1024),
        device=device,
        val_datasets=val_datasets,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    num_epochs = config.get("num_epochs", 25)
    total_steps = num_epochs * len(train_dataloader)
    scheduler = build_cosine_schedule(optimizer, total_steps, config.get("warmup_frac", 0.05))

    # --- Loss ---
    ms_loss_fn = MultiSimilarityLoss(
        alpha=config.get("ms_alpha", 2.0),
        beta=config["ms_beta"],
        lam=config["ms_lambda"],
    )

    # --- Training loop ---
    label_encoder = LabelEncoder()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    mb = config.get("micro_batch_size", 128)
    grad_clip = config.get("grad_clip", 1.0)

    for epoch in range(num_epochs):
        epoch_stats = run_train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            ms_loss_fn,
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
        avg_loss = epoch_stats["avg_loss"]

        # --- Validation ---
        eval_metrics, avg_pd_lea_f1, avg_conll_f1 = evaluate_validation(
            model, val_resources, device, threshold=EVAL_THRESHOLD,
        )

        # Report to Ray Tune (ASHA uses this to prune)
        ray_train.report({
            "epoch": epoch + 1,
            "pd_lea_f1": float(avg_pd_lea_f1),
            "conll_f1": float(avg_conll_f1),
            "train_loss": avg_loss,
        })


def run_sweep(args):
    """Configure and launch the Ray Tune sweep."""

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

    scheduler = ASHAScheduler(
        metric="pd_lea_f1",
        mode="max",
        max_t=args.num_epochs,
        grace_period=5,
        reduction_factor=3,
    )

    reporter = CLIReporter(
        metric_columns=["epoch", "pd_lea_f1", "conll_f1", "train_loss"],
        max_report_frequency=60,
    )

    trainable = tune.with_parameters(train_trial, data_dir=args.data_dir)
    trainable = tune.with_resources(trainable, {"gpu": 1})

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_trials,
        ),
        run_config=tune.RunConfig(
            name="ms_coref_sweep",
            storage_path=args.output_dir,
            progress_reporter=reporter,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("pd_lea_f1", mode="max")
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  pd_LEA F1: {best_result.metrics['pd_lea_f1']:.4f}")
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
