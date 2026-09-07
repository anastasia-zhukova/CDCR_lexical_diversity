#!/usr/bin/env python3
"""
Generate summary tables for CDCR experiment results.

Creates tables with models as rows and datasets as columns for:
  1. CoNLL F1
  2. LEA F1

Each is generated once per grouping level — eval_checkpoint.py writes both a
subtopic-level and a topic-level metrics CSV per run, so this produces four
files total: summary_{conll_f1,lea_f1}_{subtopic,topic}.csv.

Usage:
    python results_summary.py
"""

import os
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "results_only")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_only")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Friendly short names for runs
SHORT_NAMES = {
    "mix_CD2CR_CEREC_ECBplus_ECBplusMETAm_FCC_GVC_HyperCoref_MEANTIME_NIdent_NP4E_NewsWCL50_WECEng": "mix_ALL_12",
    "mix_ECBplus_ECBplusMETAm_NIdent_NewsWCL50": "mix_4",
    "mix_topic_excl_4_datasets": "mix_6_excl4",
}

# Topics to exclude for the *Excl columns
HYPERCOREF_EXCLUDE_TOPICS = {"Sport", "Sports"}
WECENG_EXCLUDE_TOPICS = {
    "Awards", "Beauty Pageant", "Concert", "Contest", "Festival", "Solar Eclipse",
}

# Canonical dataset display order
DATASET_ORDER = [
    "CD2CR", "CEREC", "ECBplus", "ECBplusMETAm", "FCC", "GVC",
    "HyperCoref", "HyperCorefExcl", "MEANTIME", "NIdent", "NP4E",
    "NewsWCL50", "WECEng", "WECEngExcl",
]


def _build_excluded_groups(level: str) -> dict[str, set[str]]:
    """Return subtopic_ids/topic_ids to exclude for HyperCoref and WECEng.

    Reads the mentions parquet to map topic -> {level}_id for the test split.

    Args:
        level: "subtopic" or "topic" — which id column to exclude on.
    """
    group_col = f"{level}_id"
    parquet_path = os.path.join(DATA_DIR, "ucdcr_mentions_processed.parquet")
    mention_df = pd.read_parquet(parquet_path)
    test_df = mention_df[mention_df["split"] == "test"]

    excluded = {}
    for dataset, excl_topics in [
        ("HyperCoref", HYPERCOREF_EXCLUDE_TOPICS),
        ("WECEng", WECENG_EXCLUDE_TOPICS),
    ]:
        ds_test = test_df[test_df["dataset"] == dataset]
        mask = ds_test["topic"].isin(excl_topics)
        excluded[dataset] = set(ds_test.loc[mask, group_col].unique())

    return excluded


def load_test_metrics(
    level: str = "subtopic", entity_type: str = "mix"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-group (subtopic or topic) test metrics for all runs.

    Args:
        level: "subtopic" (default) or "topic" — which grouping level's
            metrics CSV to read (eval_checkpoint.py writes both:
            {run}/conll_output/{setting}/test/{level}/metrics_per_{level}.csv).
        entity_type: Filter to this entity_type ('entities', 'events', 'mix').

    Returns:
        (dataset_agg, group_detail):
          - dataset_agg: DataFrame [run, dataset, conll_f1, lea_f1]
            with per-dataset macro means (including *Excl variants).
          - group_detail: DataFrame [run, dataset, {level}_id,
            conll_f1, lea_f1, n_mentions] with per-group rows
            (including *Excl virtual dataset labels).
    """
    group_col = f"{level}_id"
    excluded_groups = _build_excluded_groups(level)

    agg_frames = []
    detail_frames = []
    for csv_path in sorted(glob.glob(
        os.path.join(BASE_DIR, "*", "conll_output", "*", "test", level, f"metrics_per_{level}.csv")
    )):
        parts = csv_path.replace(BASE_DIR + "/", "").split("/")
        run_name = parts[0]

        df = pd.read_csv(csv_path)
        df = df[df["entity_type"] == entity_type]
        if df.empty:
            continue

        # Keep per-group detail
        detail = df[["dataset", group_col, "conll_f1", "lea_f1", "n_mentions"]].copy()
        detail["run"] = run_name

        # Standard per-dataset aggregation (macro over groups)
        agg = (
            df.groupby("dataset")[["conll_f1", "lea_f1"]]
            .mean()
            .reset_index()
        )
        agg["run"] = run_name

        # Compute *Excl variants
        for dataset, excl_label in [
            ("HyperCoref", "HyperCorefExcl"),
            ("WECEng", "WECEngExcl"),
        ]:
            ds_rows = df[df["dataset"] == dataset]
            if ds_rows.empty:
                continue
            ds_excl = ds_rows[~ds_rows[group_col].isin(excluded_groups[dataset])]
            if ds_excl.empty:
                continue
            excl_agg = ds_excl[["conll_f1", "lea_f1"]].mean().to_frame().T
            excl_agg["dataset"] = excl_label
            excl_agg["run"] = run_name
            agg = pd.concat([agg, excl_agg], ignore_index=True)

            # Detail rows for *Excl
            excl_detail = ds_excl[["dataset", group_col, "conll_f1", "lea_f1", "n_mentions"]].copy()
            excl_detail["dataset"] = excl_label
            excl_detail["run"] = run_name
            detail = pd.concat([detail, excl_detail], ignore_index=True)

        agg_frames.append(agg)
        detail_frames.append(detail)

    return (
        pd.concat(agg_frames, ignore_index=True),
        pd.concat(detail_frames, ignore_index=True),
    )


# Datasets used for the uCDCR summary (full versions of HyperCoref/WECEng)
_UCDCR_DATASETS = [
    "CD2CR", "CEREC", "ECBplus", "ECBplusMETAm", "FCC", "GVC",
    "HyperCoref", "MEANTIME", "NIdent", "NP4E", "NewsWCL50", "WECEng",
]
# Datasets used for the uCDCR-excl summary (*Excl versions)
_UCDCR_EXCL_DATASETS = [
    "CD2CR", "CEREC", "ECBplus", "ECBplusMETAm", "FCC", "GVC",
    "HyperCorefExcl", "MEANTIME", "NIdent", "NP4E", "NewsWCL50", "WECEngExcl",
]


def _load_run_metadata() -> dict[str, str]:
    """Return a mapping from run name (short) -> train_mode ('mix'/'single')."""
    import json
    meta = {}
    for params_path in glob.glob(os.path.join(BASE_DIR, "*", "params.json")):
        run_name = os.path.basename(os.path.dirname(params_path))
        with open(params_path) as f:
            mode = json.load(f).get("train_mode", "?")
        short = SHORT_NAMES.get(run_name, run_name)
        meta[short] = mode
    return meta


def make_pivot(
    dataset_agg: pd.DataFrame,
    group_detail: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Pivot data into a model×dataset table with macro/micro summaries.

    group_detail holds per-subtopic or per-topic rows depending on which
    level load_test_metrics() was called with — this function itself is
    level-agnostic, it only needs "dataset"/"n_mentions"/"run"/metric.
    """
    pivot = dataset_agg.pivot_table(index="run", columns="dataset", values=metric)

    # Apply short names
    pivot.index = [SHORT_NAMES.get(r, r) for r in pivot.index]

    # --- Summary columns ---
    # Macro: simple mean of per-dataset F1 values
    ucdcr_cols = [c for c in _UCDCR_DATASETS if c in pivot.columns]
    excl_cols = [c for c in _UCDCR_EXCL_DATASETS if c in pivot.columns]
    pivot["Macro"] = pivot[ucdcr_cols].mean(axis=1)
    pivot["Macro_excl"] = pivot[excl_cols].mean(axis=1)

    # Micro: mention-weighted mean across groups
    short_to_orig = {}
    for orig in group_detail["run"].unique():
        short_to_orig[SHORT_NAMES.get(orig, orig)] = orig

    micro_vals = {}
    micro_excl_vals = {}
    for short_name, orig_name in short_to_orig.items():
        run_detail = group_detail[group_detail["run"] == orig_name]

        subset = run_detail[run_detail["dataset"].isin(_UCDCR_DATASETS)]
        if not subset.empty:
            w = subset["n_mentions"]
            micro_vals[short_name] = (subset[metric] * w).sum() / w.sum()

        subset_excl = run_detail[run_detail["dataset"].isin(_UCDCR_EXCL_DATASETS)]
        if not subset_excl.empty:
            w = subset_excl["n_mentions"]
            micro_excl_vals[short_name] = (subset_excl[metric] * w).sum() / w.sum()

    pivot["Micro"] = pd.Series(micro_vals)
    pivot["Micro_excl"] = pd.Series(micro_excl_vals)

    # Add method column from params.json metadata
    run_meta = _load_run_metadata()
    pivot.insert(0, "method", pivot.index.map(lambda r: run_meta.get(r, "?")))

    # Reorder dataset columns + summaries (method stays first)
    summary_cols = ["Macro", "Macro_excl", "Micro", "Micro_excl"]
    ds_cols = [d for d in DATASET_ORDER if d in pivot.columns]
    pivot = pivot[["method"] + ds_cols + summary_cols]

    # Name the index
    pivot.index.name = "train"

    # Sort by Macro descending
    pivot = pivot.sort_values("Macro", ascending=False)

    return pivot


def highlight_best(pivot: pd.DataFrame) -> str:
    """Return a formatted string with column-wise best values marked with *."""
    numeric_cols = pivot.select_dtypes(include="number").columns
    best_idx = pivot[numeric_cols].idxmax()
    lines = []

    # Dynamic column width for numeric columns
    col_w = max(13, max(len(c) for c in pivot.columns) + 2)
    method_w = 8  # width for the method column

    # Header
    header = f"{'train':<25s}"
    for col in pivot.columns:
        if col == "method":
            header += f"{'method':>{method_w}s}"
        else:
            header += f"{col:>{col_w}s}"
    lines.append(header)
    lines.append("-" * len(header))

    for run in pivot.index:
        row = f"{run:<25s}"
        for col in pivot.columns:
            val = pivot.loc[run, col]
            if col == "method":
                row += f"{val:>{method_w}s}"
            elif pd.isna(val):
                row += f"{'':>{col_w}s}"
            else:
                marker = " *" if col in best_idx and run == best_idx[col] else "  "
                cell = f"{val:.4f}{marker}"
                row += f"{cell:>{col_w}s}"
        lines.append(row)

    return "\n".join(lines)


def main():
    for level in ("subtopic", "topic"):
        print(f"\n{'#' * 70}")
        print(f"# Level: {level}")
        print(f"{'#' * 70}\n")

        dataset_agg, group_detail = load_test_metrics(level=level, entity_type="mix")

        # --- Table 1: CoNLL F1 ---
        pivot_conll = make_pivot(dataset_agg, group_detail, "conll_f1")
        out_conll = os.path.join(OUTPUT_DIR, f"summary_conll_f1_{level}.csv")
        pivot_conll.to_csv(out_conll, float_format="%.4f")
        print(f"Wrote {out_conll}")
        print(highlight_best(pivot_conll))
        print()

        # --- Table 2: LEA F1 ---
        pivot_lea = make_pivot(dataset_agg, group_detail, "lea_f1")
        out_lea = os.path.join(OUTPUT_DIR, f"summary_lea_f1_{level}.csv")
        pivot_lea.to_csv(out_lea, float_format="%.4f")
        print(f"Wrote {out_lea}")
        print(highlight_best(pivot_lea))
        print()


if __name__ == "__main__":
    main()
