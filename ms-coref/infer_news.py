"""
NewsWCL50 inference + error analysis across val and test splits.

For each split:
  1. Threshold sweep (0.1–1.0): LEA and pd_LEA F1/P/R
  2. Deep error analysis at the best threshold:
     - Per-topic breakdown
     - Mention type breakdown (NAM / NOM / PRO)
     - NER type breakdown (PER / ORG / LOC / ...)
     - Chain-level split vs merge errors
     - Worst chains (most fragmented)

Outputs:
  --output-dir: directory for all CSV files (default: newsWCL50_analysis/)
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from coref_utils import CorefEncoder, compute_embeddings, evaluate_clustering
from sampler import MentionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.1, 0.1)]
SPLITS = ["val", "test"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, tokenizer, mlp_dim, device):
    model = CorefEncoder(tokenizer, mlp_hidden_dim=mlp_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    logger.info(
        f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
        f"best_avg_pd_lea_f1={ckpt.get('best_avg_pd_lea_f1','?')}, "
        f"best_threshold={ckpt.get('best_threshold','?')}"
    )
    return model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_split_resources(split_df, tokenizer, max_length, batch_size):
    split_df = split_df.reset_index(drop=True)
    dataset = MentionDataset(split_df, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pd_tuples = []
    for _, row in split_df.iterrows():
        head = row.get("mention_head")
        wordset = row.get("mention_wordset")
        wo_stop = row.get("mention_wo_stopwords")
        if isinstance(wordset, str) and isinstance(wo_stop, str):
            pd_tuples.append((head, wordset, wo_stop))
        else:
            pd_tuples.append((head, None, None))

    labels = split_df["coref_chain"].tolist()
    return dataloader, pd_tuples, labels


def cluster_at_threshold(embeddings, threshold):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    distances = pdist(embeddings, metric="cosine")
    Z = linkage(distances, method="average")
    return fcluster(Z, t=threshold, criterion="distance")


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_errors(split_df, embeddings, gold_labels, pd_tuples, threshold):
    """
    Run clustering at threshold and compute per-chain, per-topic,
    per-mention-type, and per-NER error stats.

    Returns a dict with DataFrames for each analysis dimension.
    """
    cluster_ids = cluster_at_threshold(embeddings, threshold)

    # Build index structures
    pred_clusters = defaultdict(list)   # pred_id -> [mention indices]
    gold_clusters = defaultdict(list)   # gold_id -> [mention indices]
    for i, (pred_id, gold_id) in enumerate(zip(cluster_ids, gold_labels)):
        pred_clusters[pred_id].append(i)
        gold_clusters[gold_id].append(i)

    mention_to_pred = {i: pred_id for pred_id, idxs in pred_clusters.items() for i in idxs}

    # -----------------------------------------------------------------------
    # Chain-level analysis
    # -----------------------------------------------------------------------
    chain_rows = []
    for gold_id, idxs in gold_clusters.items():
        pred_ids_for_chain = [mention_to_pred[i] for i in idxs]
        unique_pred = set(pred_ids_for_chain)
        dominant_pred = max(unique_pred, key=pred_ids_for_chain.count)
        purity = pred_ids_for_chain.count(dominant_pred) / len(idxs)
        split_count = len(unique_pred)

        # Check if any pred cluster for this chain contains other gold chains (merge)
        merge_count = 0
        for pid in unique_pred:
            gold_ids_in_pred = set(gold_labels[j] for j in pred_clusters[pid])
            if len(gold_ids_in_pred) > 1:
                merge_count += 1

        # Mention metadata from first mention in chain
        first = idxs[0]
        row_data = split_df.iloc[first]
        chain_rows.append({
            "chain_id": gold_id,
            "topic": row_data.get("topic", ""),
            "subtopic": row_data.get("subtopic", ""),
            "chain_size": len(idxs),
            "split_count": split_count,
            "purity": round(purity, 4),
            "merge_errors": merge_count,
            "is_split_error": split_count > 1,
            "is_merge_error": merge_count > 0,
        })

    df_chains = pd.DataFrame(chain_rows)

    # -----------------------------------------------------------------------
    # Per-topic analysis
    # -----------------------------------------------------------------------
    topic_rows = []
    for topic, tdf in df_chains.groupby("topic"):
        topic_rows.append({
            "topic": topic,
            "n_chains": len(tdf),
            "n_split_errors": tdf["is_split_error"].sum(),
            "n_merge_errors": tdf["is_merge_error"].sum(),
            "avg_split_count": round(tdf["split_count"].mean(), 2),
            "avg_purity": round(tdf["purity"].mean(), 4),
            "pct_chains_split": round(tdf["is_split_error"].mean() * 100, 1),
            "pct_chains_merged": round(tdf["is_merge_error"].mean() * 100, 1),
        })

    # Compute pd_LEA per topic by running evaluate_clustering on the topic subset
    for trow in topic_rows:
        t_mask = [gold_labels[i] in df_chains[df_chains["topic"] == trow["topic"]]["chain_id"].values
                  for i in range(len(gold_labels))]
        t_idx = [i for i, m in enumerate(t_mask) if m]
        if len(t_idx) < 2:
            trow["pd_lea_f1"] = float("nan")
            trow["lea_f1"] = float("nan")
            trow["n_mentions"] = len(t_idx)
            continue
        t_emb = embeddings[t_idx]
        t_labels = [gold_labels[i] for i in t_idx]
        t_pd = [pd_tuples[i] for i in t_idx]
        m = evaluate_clustering(t_emb, t_labels, t_pd, threshold=threshold)
        trow["n_mentions"] = len(t_idx)
        trow["pd_lea_f1"] = round(m["pd_lea_f1"], 4)
        trow["lea_f1"] = round(m["lea_f1"], 4)

    df_topics = pd.DataFrame(topic_rows).sort_values("pd_lea_f1")

    # -----------------------------------------------------------------------
    # Per-mention-type analysis
    # -----------------------------------------------------------------------
    type_rows = []
    for mtype, grp in split_df.groupby("mention_type"):
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue
        t_emb = embeddings[idxs]
        t_labels = [gold_labels[i] for i in idxs]
        t_pd = [pd_tuples[i] for i in idxs]
        m = evaluate_clustering(t_emb, t_labels, t_pd, threshold=threshold)

        # Count split/merge errors for mentions of this type
        split_err = sum(
            1 for i in idxs
            if len(set(mention_to_pred[j] for j in gold_clusters[gold_labels[i]])) > 1
        )
        type_rows.append({
            "mention_type": mtype,
            "n_mentions": len(idxs),
            "pd_lea_f1": round(m["pd_lea_f1"], 4),
            "lea_f1": round(m["lea_f1"], 4),
            "pd_lea_p": round(m["pd_lea_precision"], 4),
            "pd_lea_r": round(m["pd_lea_recall"], 4),
            "n_split_errors": split_err,
            "pct_in_split_chains": round(split_err / len(idxs) * 100, 1),
        })
    df_mtypes = pd.DataFrame(type_rows).sort_values("pd_lea_f1")

    # -----------------------------------------------------------------------
    # Per-NER type (NAM mentions only)
    # -----------------------------------------------------------------------
    nam_df = split_df[split_df["mention_type"] == "NAM"]
    ner_rows = []
    for ner, grp in nam_df.groupby("mention_ner"):
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue
        t_emb = embeddings[idxs]
        t_labels = [gold_labels[i] for i in idxs]
        t_pd = [pd_tuples[i] for i in idxs]
        m = evaluate_clustering(t_emb, t_labels, t_pd, threshold=threshold)
        ner_rows.append({
            "ner_type": ner,
            "n_mentions": len(idxs),
            "pd_lea_f1": round(m["pd_lea_f1"], 4),
            "lea_f1": round(m["lea_f1"], 4),
            "pd_lea_p": round(m["pd_lea_precision"], 4),
            "pd_lea_r": round(m["pd_lea_recall"], 4),
        })
    df_ner = pd.DataFrame(ner_rows).sort_values("pd_lea_f1") if ner_rows else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Worst chains (most fragmented)
    # -----------------------------------------------------------------------
    df_worst = (
        df_chains[df_chains["chain_size"] > 1]
        .sort_values(["split_count", "purity"], ascending=[False, True])
        .head(20)
    )
    # Add head words for context
    head_map = {}
    for gold_id, idxs in gold_clusters.items():
        heads = split_df.iloc[idxs]["mention_head"].dropna().tolist()
        head_map[gold_id] = ", ".join(sorted(set(str(h) for h in heads))[:8])
    df_worst = df_worst.copy()
    df_worst["head_words"] = df_worst["chain_id"].map(head_map)

    return {
        "chains": df_chains,
        "topics": df_topics,
        "mention_types": df_mtypes,
        "ner_types": df_ner,
        "worst_chains": df_worst,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_threshold_table(split, results, best_threshold):
    col_w = 9
    print(f"\n{'='*90}")
    print(f"NewsWCL50 — {split.upper()} split  |  Threshold sweep")
    print(f"{'='*90}")
    header = (f"  {'thresh':>6}  {'LEA_P':>{col_w}}{'LEA_R':>{col_w}}{'LEA_F1':>{col_w}}  "
              f"{'pdLEA_P':>{col_w}}{'pdLEA_R':>{col_w}}{'pdLEA_F1':>{col_w}}  "
              f"{'gold':>6}{'pred':>6}")
    print(header)
    print("-" * len(header))
    for t, m in results.items():
        marker = "*" if t == best_threshold else " "
        print(
            f"  {t:.1f} {marker}  "
            f"{m['lea_precision']:>{col_w}.4f}{m['lea_recall']:>{col_w}.4f}{m['lea_f1']:>{col_w}.4f}  "
            f"{m['pd_lea_precision']:>{col_w}.4f}{m['pd_lea_recall']:>{col_w}.4f}{m['pd_lea_f1']:>{col_w}.4f}  "
            f"{m['n_gold_clusters']:>6}{m['n_pred_clusters']:>6}"
        )
    print(f"\n  Best threshold: {best_threshold:.1f}  "
          f"(pd_LEA F1={results[best_threshold]['pd_lea_f1']:.4f}  "
          f"LEA F1={results[best_threshold]['lea_f1']:.4f})")


def print_error_analysis(analysis, best_threshold):
    print(f"\n{'='*90}")
    print(f"Error analysis at threshold {best_threshold:.1f}")
    print(f"{'='*90}")

    # Topics
    print("\n-- Per-Topic (sorted by pd_LEA F1 asc) --")
    df = analysis["topics"]
    cols = ["topic", "n_mentions", "n_chains", "pd_lea_f1", "lea_f1",
            "pct_chains_split", "pct_chains_merged", "avg_purity"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Mention types
    print("\n-- Mention Type --")
    df = analysis["mention_types"]
    cols = ["mention_type", "n_mentions", "pd_lea_f1", "lea_f1",
            "pd_lea_p", "pd_lea_r", "pct_in_split_chains"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # NER types
    if not analysis["ner_types"].empty:
        print("\n-- NER Type (NAM mentions only) --")
        df = analysis["ner_types"]
        cols = ["ner_type", "n_mentions", "pd_lea_f1", "lea_f1", "pd_lea_p", "pd_lea_r"]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Error summary
    df_chains = analysis["chains"]
    n_total = len(df_chains[df_chains["chain_size"] > 1])
    n_split = df_chains["is_split_error"].sum()
    n_merge = df_chains["is_merge_error"].sum()
    print(f"\n-- Overall chain errors (non-singleton chains only) --")
    print(f"  Total non-singleton chains : {n_total}")
    print(f"  Split errors (fragmented)  : {n_split} ({100*n_split/max(n_total,1):.1f}%)")
    print(f"  Merge errors (over-merged) : {n_merge} ({100*n_merge/max(n_total,1):.1f}%)")

    # Worst chains
    print("\n-- Worst chains (most fragmented) --")
    df = analysis["worst_chains"][
        ["chain_id", "topic", "chain_size", "split_count", "purity",
         "merge_errors", "head_words"]
    ]
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    mention_df = pd.read_parquet(Path(args.data_dir) / "ucdcr_mentions_processed.parquet")
    news_df = mention_df[mention_df["dataset"] == "NewsWCL50"].copy()
    logger.info(f"NewsWCL50: {len(news_df):,} mentions, splits: {sorted(news_df['split'].unique())}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])
    model = load_model(args.checkpoint, tokenizer, args.mlp_dim, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sweep_rows = []

    for split in SPLITS:
        split_df = news_df[news_df["split"].isin(
            [split, "validation"] if split == "val" else [split]
        )]
        if split_df.empty:
            logger.warning(f"No mentions for split '{split}', skipping")
            continue

        logger.info(f"\n--- {split.upper()} split: {len(split_df):,} mentions ---")
        split_df = split_df.reset_index(drop=True)
        dataloader, pd_tuples, gold_labels = build_split_resources(
            split_df, tokenizer, args.max_length, args.batch_size
        )

        logger.info("Computing embeddings...")
        embeddings, _ = compute_embeddings(model, dataloader, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Threshold sweep
        sweep_results = {}
        for t in THRESHOLDS:
            sweep_results[t] = evaluate_clustering(embeddings, gold_labels, pd_tuples, threshold=t)

        best_threshold = max(sweep_results, key=lambda t, r=sweep_results: r[t]["pd_lea_f1"])
        print_threshold_table(split, sweep_results, best_threshold)

        for t, m in sweep_results.items():
            all_sweep_rows.append({"split": split, "threshold": t, **m})

        # Error analysis at best threshold
        logger.info(f"Running error analysis at threshold {best_threshold:.1f}...")
        analysis = analyze_errors(split_df, embeddings, gold_labels, pd_tuples, best_threshold)
        print_error_analysis(analysis, best_threshold)

        # Save per-split CSVs
        for name, df in analysis.items():
            path = out_dir / f"{split}_{name}.csv"
            df.to_csv(path, index=False, float_format="%.4f")
            logger.info(f"  Saved {path}")

    # Save sweep CSV
    sweep_path = out_dir / "threshold_sweep.csv"
    pd.DataFrame(all_sweep_rows).to_csv(sweep_path, index=False, float_format="%.4f")
    logger.info(f"Sweep results saved to {sweep_path}")


def _load_config():
    base = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    cfg = _load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=cfg.get("checkpoint"))
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="newsWCL50_analysis")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
