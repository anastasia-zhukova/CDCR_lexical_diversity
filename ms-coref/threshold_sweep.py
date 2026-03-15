"""
Clustering threshold sweep for MS-coref.

Loads a trained checkpoint, computes embeddings once per validation dataset,
then evaluates pd_LEA F1 across cosine-distance thresholds 0.1–0.9.

Prints a table: rows = thresholds, columns = datasets + avg.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer

from coref_utils import CorefEncoder, compute_embeddings, evaluate_clustering
from train import build_val_resources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]


def load_model(checkpoint_path, tokenizer, mlp_dim, device):
    model = CorefEncoder(tokenizer, mlp_hidden_dim=mlp_dim)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_f1 = ckpt.get("best_avg_pd_lea_f1", ckpt.get("best_pd_lea_f1", "?"))
    logger.info(f"Loaded checkpoint: epoch={epoch}, best_avg_pd_lea_f1={best_f1}")
    return model


def sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Data ---
    data_path = Path(args.data_dir) / "ucdcr_mentions_processed.parquet"
    mention_df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(mention_df):,} mentions")

    # --- Tokenizer + model ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])

    model = load_model(args.checkpoint, tokenizer, args.mlp_dim, device)

    # --- Validation resources ---
    val_resources = build_val_resources(
        mention_df, tokenizer, args.max_length, args.val_batch_size, args.num_workers
    )
    dataset_names = sorted(val_resources.keys())
    logger.info(f"Datasets: {dataset_names}")

    # --- Compute embeddings once per dataset ---
    logger.info("Computing embeddings for all validation datasets...")
    embeddings_cache = {}
    for ds_name in dataset_names:
        res = val_resources[ds_name]
        embs, labels = compute_embeddings(model, res["dataloader"], device)
        embeddings_cache[ds_name] = (embs, labels, res["pd_tuples"])
        logger.info(f"  {ds_name}: {embs.shape[0]} mentions embedded")

    # --- Sweep thresholds ---
    logger.info(f"Sweeping thresholds: {THRESHOLDS}")
    results = {}  # threshold -> {dataset -> pd_lea_f1}

    for t in THRESHOLDS:
        results[t] = {}
        for ds_name in dataset_names:
            embs, labels, pd_tuples = embeddings_cache[ds_name]
            metrics = evaluate_clustering(embs, labels, pd_tuples, threshold=t)
            results[t][ds_name] = metrics["pd_lea_f1"]

    # --- Build results table ---
    rows = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        ds_vals = [results[t][ds] for ds in dataset_names]
        for ds, v in zip(dataset_names, ds_vals):
            row[ds] = v
        row["avg_pd_lea_f1"] = np.mean(ds_vals)
        rows.append(row)

    df_results = pd.DataFrame(rows).set_index("threshold")

    # --- Print table ---
    print("\n" + "=" * 80)
    print("pd_LEA F1 by clustering threshold")
    print("=" * 80)

    # Header
    col_w = 14
    header = f"{'thresh':>7}  " + "".join(f"{ds[:col_w]:>{col_w}}" for ds in dataset_names)
    header += f"{'AVG':>{col_w}}"
    print(header)
    print("-" * len(header))

    best_avg = df_results["avg_pd_lea_f1"].max()
    for t, row in df_results.iterrows():
        marker = " *" if abs(row["avg_pd_lea_f1"] - best_avg) < 1e-9 else "  "
        line = f"  {t:.1f}  {marker}" + "".join(
            f"{row[ds]:>{col_w}.4f}" for ds in dataset_names
        )
        line += f"{row['avg_pd_lea_f1']:>{col_w}.4f}"
        print(line)

    print("-" * len(header))
    best_thresh = df_results["avg_pd_lea_f1"].idxmax()
    print(f"\nBest threshold: {best_thresh:.1f}  (avg pd_LEA F1 = {best_avg:.4f})")

    # --- Save CSV ---
    out_path = Path(args.output_dir) / "threshold_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, float_format="%.4f")
    logger.info(f"Results saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Threshold sweep for MS-coref clustering")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sweep(args)
