"""
Full CoNLL Evaluation — All Checkpoints

Walks RESULTS_ROOT, finds checkpoint directories, runs eval_checkpoint.py on
all 12 CDCR datasets (test split) at both the subtopic and topic grouping
levels, and writes output inside each run directory as <run_dir>/conll_output/.

Usage:
    python eval_all_checkpoints.py
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
import yaml


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()

RESULTS_ROOT = CONFIG["results_root"]
DATA_DIR = CONFIG["data_dir"]
REPO_DIR = CONFIG["repo_dir"]
SPLIT = CONFIG["split"]
BATCH_SIZE = CONFIG["batch_size"]
MAX_LENGTH = CONFIG["max_length"]
MLP_DIM = CONFIG["mlp_dim"]
ALL_DATASETS = CONFIG["all_datasets"]
EXCLUDE_TOPICS = CONFIG["exclude_topics"]

# Both grouping levels are always evaluated in one pass (see eval_checkpoint.py).
LEVELS = ("subtopic", "topic")


def _find_checkpoint(run_dir: Path):
    """Return the best-model weights checkpoint inside run_dir/checkpoints/."""
    ckpt_dir = run_dir / "checkpoints"
    for name in ("best_model_pd_lea_weights.pt", "best_model_weights.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    return None


def discover_runs(results_root: Path):
    """Discover all run directories under results_root that contain a checkpoint."""
    runs = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpt = _find_checkpoint(run_dir)
        if ckpt is None:
            print(f"WARNING: no checkpoint found in {run_dir.name} — skipping")
            continue
        runs.append({"run_dir": run_dir, "run_name": run_dir.name, "checkpoint": ckpt})

    print(f"Discovered {len(runs)} runs")
    for r in runs:
        print(f"  {r['run_name']}  →  {r['checkpoint'].name}")
    return runs


def evaluate_runs(runs, data_dir: str, split: str, batch_size: int):
    """Run eval_checkpoint on each discovered run, collecting per-level metrics."""
    import pandas as pd

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    import eval_checkpoint as _ce

    results_summary = {level: [] for level in LEVELS}
    failed_runs = []

    for i, run in enumerate(runs):
        run_name = run["run_name"]
        output_dir = run["run_dir"] / "conll_output"
        output_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(runs)}]  {run_name}  ({ts})")
        print(f"  checkpoint:  {run['checkpoint']}")
        print(f"  output_dir:  {output_dir}")

        try:
            args = argparse.Namespace(
                checkpoint=str(run["checkpoint"]),
                data_dir=data_dir,
                output_dir=str(output_dir),
                split=split,
                setting=run_name,
                max_length=MAX_LENGTH,
                mlp_dim=MLP_DIM,
                batch_size=batch_size,
                datasets=ALL_DATASETS,
                exclude_topics=EXCLUDE_TOPICS,
            )
            _ce.run(args)

            run_out_dir = output_dir / run_name / split
            for level in LEVELS:
                csv_path = run_out_dir / level / f"metrics_per_{level}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    avg_conll = round(df["conll_f1"].mean(), 4)
                    avg_lea = round(df["lea_f1"].mean(), 4)
                    results_summary[level].append(dict(
                        run_name=run_name,
                        n_groups=len(df),
                        avg_conll_f1=avg_conll,
                        avg_lea_f1=avg_lea,
                        status="OK",
                    ))
                    print(f"  [{level}] OK  avg CoNLL F1={avg_conll:.4f}  "
                          f"avg LEA F1={avg_lea:.4f}  ({len(df)} groups)")
                else:
                    results_summary[level].append(dict(run_name=run_name, status="CSV missing"))
                    print(f"  [{level}] WARN: metrics_per_{level}.csv not found")

            stats_path = run_out_dir / "eval_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                print(f"  timing: {stats['total_time_sec']}s total "
                      f"({stats['embedding_time_sec']}s embedding, "
                      f"{stats['mentions_per_sec']} mentions/s)")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            failed_runs.append(run_name)
            for level in LEVELS:
                results_summary[level].append(dict(run_name=run_name, status=f"FAILED: {exc}"))

    print(f"\n{'='*70}")
    print(f"Finished: {len(runs) - len(failed_runs)}/{len(runs)} runs succeeded.")
    if failed_runs:
        print(f"Failed: {failed_runs}")

    return results_summary


def save_summary(results_summary, results_root: Path, split: str):
    """Save the per-level, per-run summaries as CSVs and return the DataFrames."""
    import pandas as pd

    summaries = {}
    for level, rows in results_summary.items():
        summary_df = pd.DataFrame(rows)
        summary_csv = results_root / f"eval_all_checkpoints_summary_{split}_{level}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary ({level}) written to {summary_csv}")
        summaries[level] = summary_df
    return summaries


def main():
    results_root = Path(RESULTS_ROOT)
    runs = discover_runs(results_root)
    results_summary = evaluate_runs(runs, DATA_DIR, SPLIT, BATCH_SIZE)
    save_summary(results_summary, results_root, SPLIT)


if __name__ == "__main__":
    main()
