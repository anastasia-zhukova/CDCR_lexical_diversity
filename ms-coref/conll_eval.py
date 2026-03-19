"""
CoNLL file generation and metric CSV output for cross-checking with Anastasia's Perl scorer.

Directory structure:
    {output_dir}/{setting}/{split}/{dataset}/{entity_type}/{subtopic_id}/
        key.conll      — gold clusters (integer chain IDs in brackets)
        response.conll — predicted clusters (integer cluster IDs in brackets)

  - Clustering is performed per subtopic, independently for each entity_type
  - entity_type is one of: "entities" (NAM/NOM/PRO), "events" (all other types), "mix" (all)
  - "mix" is created for every subtopic with >= 2 non-singleton mentions (all types combined)
  - Singletons are excluded (per Anastasia's request)
  - Coref chain names are converted to integers (required by Perl scorer)
  - Each file contains exactly one document (the subtopic)

Also outputs a CSV with per-subtopic metrics so Anastasia can aggregate and
compare against her Perl scorer output.

CSV columns:
    setting, dataset, entity_type, subtopic_id,
    muc_p, muc_r, muc_f1,
    b3_p,  b3_r,  b3_f1,
    ceafe_p, ceafe_r, ceafe_f1,
    conll_f1,
    lea_p, lea_r, lea_f1,
    n_mentions, n_chains_gold, n_chains_pred

Usage:
    python conll_eval.py --checkpoint /path/to/best_model.pt --split test
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from coref_utils import CorefEncoder, compute_embeddings
from pk_sampler import MentionDataset
from train import EVAL_THRESHOLD

ALL_DATASETS = ["HyperCoref", "NP4E", "NewsWCL50", "NIdent", "ECBplus", "ECBplusMETAm"]

# Mirror the exclusion from train.py — set to None or empty set to disable
HYPERCOREF_EXCLUDE_TOPICS = {"Sport", "Sports"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Event-mention detection via mention_full_type.
# mention_type is NOT used — it is dataset-specific (e.g. "ACT" means
# "Actor/person" in NewsWCL50 but "action event" in ECBplus).
#
# Rules per dataset:
#   ECBplus / ECBplusMETAm : ACTION_* and NEG_ACTION_* → event
#   NewsWCL50              : ACTION and EVENT → event
#   HyperCoref             : EVENT → event  (Wikipedia event-article hyperlinks)
#   NIdent / NP4E          : EVENT → entity (named event NER category, not event coref)
#
# Implemented as: event = contains("ACTION") OR
#                         (== "EVENT" AND dataset not in _EVENT_IS_ENTITY_DATASETS)
_EVENT_IS_ENTITY_DATASETS = frozenset({"NIdent", "NP4E"})

SETTING = "mixed_datasets"


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
    epoch = ckpt.get("epoch", "?")
    best_f1 = ckpt.get("best_avg_pd_lea_f1", "?")
    logger.info(f"Loaded checkpoint: epoch={epoch}, best_avg_pd_lea_f1={best_f1}")
    return model


# ---------------------------------------------------------------------------
# Metrics (inline, no coref_utils dependency for per-subtopic slicing)
# ---------------------------------------------------------------------------

def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _muc(clusters, mention_to_other):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = {id(mention_to_other[m]) for m in c if m in mention_to_other}
        tp -= len(linked)
    return tp, p


def _b3(clusters, mention_to_other_cluster):
    from collections import Counter
    num, den = 0.0, 0.0
    for c in clusters:
        counts = Counter(id(mention_to_other_cluster[m]) for m in c if m in mention_to_other_cluster)
        correct = sum(v * v for v in counts.values())
        num += correct / len(c)
        den += len(c)
    return num, den


def _ceafe(pred_list, gold_list):
    from scipy.optimize import linear_sum_assignment
    scores = np.zeros((len(gold_list), len(pred_list)))
    for i, gc in enumerate(gold_list):
        for j, pc in enumerate(pred_list):
            overlap = len(gc & pc)
            scores[i, j] = 2 * overlap / (len(gc) + len(pc)) if (len(gc) + len(pc)) > 0 else 0
    row_ind, col_ind = linear_sum_assignment(-scores)
    sim = scores[row_ind, col_ind].sum()
    return sim, len(pred_list), sim, len(gold_list)


def compute_metrics_for_clusters(pred_clusters, gold_clusters):
    """
    Given pred_clusters and gold_clusters (both dicts of id -> set of indices),
    compute MUC, B³, CEAFe, LEA and CoNLL F1.
    """
    pred_list = list(pred_clusters.values())
    gold_list = list(gold_clusters.values())

    mention_to_gold = {m: c for c in gold_list for m in c}
    mention_to_pred = {m: c for c in pred_list for m in c}

    # MUC
    mp_n, mp_d = _muc(pred_list, mention_to_gold)
    mr_n, mr_d = _muc(gold_list, mention_to_pred)
    muc_p = mp_n / mp_d if mp_d > 0 else 0
    muc_r = mr_n / mr_d if mr_d > 0 else 0
    muc_f1 = _f1(muc_p, muc_r)

    # B³
    bp_n, bp_d = _b3(pred_list, mention_to_gold)
    br_n, br_d = _b3(gold_list, mention_to_pred)
    b3_p = bp_n / bp_d if bp_d > 0 else 0
    b3_r = br_n / br_d if br_d > 0 else 0
    b3_f1 = _f1(b3_p, b3_r)

    # CEAFe
    ep_n, ep_d, er_n, er_d = _ceafe(pred_list, gold_list)
    ceafe_p = ep_n / ep_d if ep_d > 0 else 0
    ceafe_r = er_n / er_d if er_d > 0 else 0
    ceafe_f1 = _f1(ceafe_p, ceafe_r)

    conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3

    # LEA
    def _lea_num_den(inp, m2o):
        num, den = 0.0, 0.0
        for c in inp:
            den += len(c)
            if len(c) == 1:
                m = next(iter(c))
                if len(m2o.get(m, set())) == 1:
                    num += 1
            else:
                all_links = len(c) * (len(c) - 1) / 2.0
                common = sum(len(c & m2o.get(m, set())) - 1 for m in c) / 2.0
                num += len(c) * (common / all_links) if all_links > 0 else 0
        return num, den

    lp_n, lp_d = _lea_num_den(pred_list, mention_to_gold)
    lr_n, lr_d = _lea_num_den(gold_list, mention_to_pred)
    lea_p = lp_n / lp_d if lp_d > 0 else 0
    lea_r = lr_n / lr_d if lr_d > 0 else 0
    lea_f1 = _f1(lea_p, lea_r)

    return {
        "muc_p": muc_p, "muc_r": muc_r, "muc_f1": muc_f1,
        "b3_p": b3_p, "b3_r": b3_r, "b3_f1": b3_f1,
        "ceafe_p": ceafe_p, "ceafe_r": ceafe_r, "ceafe_f1": ceafe_f1,
        "conll_f1": conll_f1,
        "lea_p": lea_p, "lea_r": lea_r, "lea_f1": lea_f1,
        "n_chains_gold": len(gold_list),
        "n_chains_pred": len(pred_list),
    }


# ---------------------------------------------------------------------------
# Entity-type splitting
# ---------------------------------------------------------------------------

def _is_event(full_type_series, dataset=""):
    """
    Boolean Series: True if the mention is an event mention.

      - contains "ACTION" → event for all datasets (ACTION_*, NEG_ACTION_*, ACTION)
      - == "EVENT"        → event unless dataset is in _EVENT_IS_ENTITY_DATASETS
                            (NIdent/NP4E use EVENT as a named-entity NER category)
    """
    fts = full_type_series.fillna("")
    is_action = fts.str.contains("ACTION", case=False, na=False)
    is_event_type = fts.eq("EVENT") & (dataset not in _EVENT_IS_ENTITY_DATASETS)
    return is_action | is_event_type


def get_entity_type_subsets(st_df):
    """
    Split a subtopic DataFrame into entity_type subsets for separate clustering.

    Returns a dict of entity_type -> DataFrame (non-singleton mentions only):
      - "entities": non-event mentions (HUMAN_*, LOC_*, ACTOR, COUNTRY, PERSON, …)
      - "events":   event mentions (EVENT, ACTION_*, NEG_ACTION_*, ACTION)
      - "mix":      all non-singleton mentions — whenever >= 2 total non-singletons exist

    Types with fewer than 2 mentions are excluded (cannot form clusters).
    """
    non_single = st_df[~st_df["is_singleton"]]
    dataset = st_df["dataset"].iloc[0] if len(st_df) > 0 else ""
    is_ev = _is_event(non_single["mention_full_type"], dataset)
    entity_df = non_single[~is_ev]
    event_df = non_single[is_ev]

    has_entities = len(entity_df) >= 2
    has_events = len(event_df) >= 2

    result = {}
    if has_entities:
        result["entities"] = entity_df
    if has_events:
        result["events"] = event_df
    if len(non_single) >= 2:
        result["mix"] = non_single
    return result


# ---------------------------------------------------------------------------
# CoNLL writing
# ---------------------------------------------------------------------------

def write_conll_subtopic(mentions_df, cluster_id_col, gold_chain_col,
                         chain_to_int, dataset, entity_type, subtopic_id, out_dir):
    """
    Write key.conll (gold) and response.conll (predicted) for one
    (dataset, entity_type, subtopic_id) combination.

    Each file contains exactly one document:
        #begin document (dataset/entity_type/subtopic_id); part 000
        doc_id  mention_id  (integer_gold_chain)   — key
        doc_id  mention_id  (integer_pred_cluster) — response
        #end document
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path = out_dir / "key.conll"
    sys_path = out_dir / "response.conll"

    doc_id = f"{dataset}/{entity_type}/{subtopic_id}"
    header = f"#begin document ({doc_id}); part 000\n"

    with open(key_path, "w") as kf, open(sys_path, "w") as sf:
        kf.write(header)
        sf.write(header)

        for _, row in mentions_df.iterrows():
            mid = row["mention_id"]
            gold_int = chain_to_int[row[gold_chain_col]]
            pred_int = int(row[cluster_id_col])
            kf.write(f"{doc_id}\t{mid}\t({gold_int})\n")
            sf.write(f"{doc_id}\t{mid}\t({pred_int})\n")

        kf.write("#end document\n\n")
        sf.write("#end document\n\n")

    return key_path, sys_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cluster_subset(embs, pos_indices, threshold):
    """Cluster a subset of embeddings (by positional index) at the given threshold."""
    sub_embs = embs[pos_indices]
    distances = pdist(sub_embs, metric="cosine")
    Z = linkage(distances, method="average")
    return fcluster(Z, t=threshold, criterion="distance")


def _metrics_from_df(sub_df):
    """Compute coreference metrics from a DataFrame with _pred_cluster and coref_chain."""
    pred_clusters = defaultdict(set)
    gold_clusters = defaultdict(set)
    for local_i, (_, row) in enumerate(sub_df.iterrows()):
        pred_clusters[int(row["_pred_cluster"])].add(local_i)
        gold_clusters[row["coref_chain"]].add(local_i)
    return pred_clusters, gold_clusters


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    mention_df = pd.read_parquet(Path(args.data_dir) / "ucdcr_mentions_processed.parquet")
    mention_df = mention_df[mention_df["dataset"].isin(args.datasets)].copy()

    # Exclude HyperCoref sport topics (mirrors train.py exclusion)
    if HYPERCOREF_EXCLUDE_TOPICS:
        mask = (mention_df["dataset"] == "HyperCoref") & \
               (mention_df["topic"].isin(HYPERCOREF_EXCLUDE_TOPICS))
        n_excluded = mask.sum()
        mention_df = mention_df[~mask].copy()
        logger.info(f"Excluded {n_excluded:,} HyperCoref mentions with topics "
                    f"{HYPERCOREF_EXCLUDE_TOPICS}")

    # Map splits: "val" also matches "validation"
    split_values = (["val", "validation"] if args.split == "val"
                    else [args.split])
    split_df = mention_df[mention_df["split"].isin(split_values)].copy()
    logger.info(f"Split '{args.split}': {len(split_df):,} mentions "
                f"across {split_df['dataset'].nunique()} datasets")

    # Load model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenizer.add_tokens(["[E]", "[/E]"])
    model = load_model(args.checkpoint, tokenizer, args.mlp_dim, device)

    setting = args.setting

    out_dir = Path(args.output_dir) / setting / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate entity/event split across all datasets (per-dataset rules applied)
    logger.info("\n--- mention_full_type entity/event split validation ---")
    for ds in sorted(split_df["dataset"].unique()):
        mask = split_df["dataset"] == ds
        is_ev = _is_event(split_df.loc[mask, "mention_full_type"], ds)
        n_entity = (~is_ev).sum()
        n_event = is_ev.sum()
        n_null = split_df.loc[mask, "mention_full_type"].isna().sum()
        logger.info(f"  {ds:20s}  entities={n_entity:,}  events={n_event:,}  null={n_null:,}")
    logger.info("------------------------------------------------------\n")

    all_rows = []

    for dataset in sorted(split_df["dataset"].unique()):
        # Reset index so positional index aligns with embedding array
        ds_df = split_df[split_df["dataset"] == dataset].reset_index(drop=True)
        logger.info(f"\n{dataset}: {len(ds_df):,} mentions")

        # Compute embeddings once for all mentions in this dataset
        dataset_obj = MentionDataset(ds_df, tokenizer, max_length=args.max_length)
        loader = DataLoader(dataset_obj, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        embs, _ = compute_embeddings(model, loader, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ds_conll_dir = out_dir / dataset
        entity_type_counts = defaultdict(int)

        for subtopic_id, st_df in ds_df.groupby("subtopic_id", sort=True):
            # st_df.index contains positional indices into embs (ds_df was reset_index'd)
            subsets = get_entity_type_subsets(st_df)
            if not subsets:
                continue

            for entity_type, sub_df in subsets.items():
                pos_indices = sub_df.index.tolist()
                cluster_ids = _cluster_subset(embs, pos_indices, EVAL_THRESHOLD)

                sub_df = sub_df.copy()
                sub_df["_pred_cluster"] = cluster_ids

                # Per-subtopic chain → int mapping (Perl scorer needs integers)
                unique_chains = sorted(sub_df["coref_chain"].unique())
                chain_to_int = {c: i + 1 for i, c in enumerate(unique_chains)}

                # Write CoNLL files: {out_dir}/{dataset}/{entity_type}/{subtopic_id}/
                conll_dir = ds_conll_dir / entity_type / str(subtopic_id)
                write_conll_subtopic(
                    mentions_df=sub_df,
                    cluster_id_col="_pred_cluster",
                    gold_chain_col="coref_chain",
                    chain_to_int=chain_to_int,
                    dataset=dataset,
                    entity_type=entity_type,
                    subtopic_id=subtopic_id,
                    out_dir=conll_dir,
                )

                # Metrics for CSV (skip if all gold chains are singletons)
                pred_clusters, gold_clusters = _metrics_from_df(sub_df)
                if all(len(c) == 1 for c in gold_clusters.values()):
                    continue

                m = compute_metrics_for_clusters(pred_clusters, gold_clusters)
                all_rows.append({
                    "setting": setting,
                    "dataset": dataset,
                    "entity_type": entity_type,
                    "subtopic_id": subtopic_id,
                    "n_mentions": len(sub_df),
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in m.items()},
                })
                entity_type_counts[entity_type] += 1

        # Dataset-level summary: log subtopic counts per entity_type
        for entity_type, n_subtopics in sorted(entity_type_counts.items()):
            logger.info(f"  {entity_type:8s}  {n_subtopics} subtopics written")
        logger.info(f"  CoNLL files → {ds_conll_dir}")

    # Save CSV
    csv_path = out_dir / "metrics_per_subtopic.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"\nMetrics CSV saved to {csv_path} ({len(all_rows)} rows)")


def parse_args():
    p = argparse.ArgumentParser(description="CoNLL file generation + metric CSV")
    p.add_argument("--checkpoint", type=str,
                   default="/Volumes/jnj_polyphonic_deid/default/cdcr/ms_coref_v2/checkpoints/best_model.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="conll_output")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--setting", type=str, default=SETTING,
                   help="Experiment name for output directory and CSV")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   help="Datasets to evaluate on (default: all 6)")
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
