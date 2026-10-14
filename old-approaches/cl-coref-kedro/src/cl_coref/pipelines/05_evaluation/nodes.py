"""
Nodes for evaluation pipeline.

This module evaluates CoNLL inference results and computes comprehensive metrics.
"""

import logging
import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def parse_conll_file(conll_path: str) -> Dict[str, str]:
    """
    Parse a CoNLL file and extract cluster assignments.

    Args:
        conll_path: Path to CoNLL file

    Returns:
        Dictionary mapping (doc_id, line_number) to cluster_id
    """
    cluster_map = {}
    current_doc = None
    line_number = 0

    with open(conll_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('#begin document'):
                # Extract document ID
                # Format: #begin document (dataset/subtopic); part 000
                current_doc = line.split('(')[1].split(')')[0]
                line_number = 0
                continue

            # Parse mention line
            # Format: dataset/subtopic    (cluster_id)
            # Split on whitespace (tabs or spaces)
            parts = line.split()
            if len(parts) >= 2:
                cluster_str = parts[-1].strip()
                # Extract cluster ID from parentheses
                if cluster_str.startswith('(') and cluster_str.endswith(')'):
                    cluster_id = cluster_str[1:-1]
                    cluster_map[(current_doc, line_number)] = cluster_id
                    line_number += 1

    return cluster_map


def load_mentions_with_metadata(
    mentions_df: pd.DataFrame,
    training_config: str,
    dataset_name: str,
    mode: str
) -> pd.DataFrame:
    """
    Load and filter mentions for a specific evaluation configuration.

    Args:
        mentions_df: Full mentions DataFrame
        training_config: Training configuration name
        dataset_name: Test dataset name
        mode: Evaluation mode (Events, Entities, Mix)

    Returns:
        Filtered mentions DataFrame with required metadata
    """
    # Filter to test split and specific dataset
    filtered = mentions_df[
        (mentions_df['split'] == 'test') &
        (mentions_df['dataset_name'] == dataset_name)
    ].copy()

    # Apply mode filter
    if mode == 'Events':
        filtered = filtered[filtered['entity_style'] == 'event']
    elif mode == 'Entities':
        filtered = filtered[filtered['entity_style'] == 'entity']
    # Mix mode keeps all mentions

    # Sort by subtopic_id and mention_id to match CoNLL file order
    filtered = filtered.sort_values(
        ['subtopic_id', 'mention_id']
    ).reset_index(drop=True)

    return filtered


def evaluate_single_configuration(
    mentions_df: pd.DataFrame,
    training_config: str,
    dataset_name: str,
    mode: str,
    key_conll_path: str,
    response_conll_path: str
) -> pd.DataFrame:
    """
    Evaluate a single configuration using CoNLL files.

    Uses the official coval library directly on CoNLL files,
    bypassing the need to reconstruct predictions_df format.

    Args:
        mentions_df: Mentions DataFrame with metadata
        training_config: Training configuration name
        dataset_name: Test dataset name
        mode: Evaluation mode (Events, Entities, Mix)
        key_conll_path: Path to gold standard CoNLL file
        response_conll_path: Path to predictions CoNLL file

    Returns:
        DataFrame with evaluation metrics
    """
    logger.info(f"Evaluating: {training_config} / {dataset_name} / {mode}")

    # Import coval directly for evaluation
    CL_COREF_PATH = Path(__file__).parents[5] / "cl-coref-legacy"
    sys.path.insert(0, str(CL_COREF_PATH))

    from coval.conll import reader
    from coval.eval import evaluator

    # Count mentions for reporting
    filtered_mentions = load_mentions_with_metadata(
        mentions_df, training_config, dataset_name, mode
    )

    if len(filtered_mentions) == 0:
        logger.warning(f"No mentions found for {dataset_name} / {mode}")
        return None

    num_mentions = len(filtered_mentions)

    try:
        # Use coval's reader to parse CoNLL files directly
        doc_coref_infos = reader.get_coref_infos(
            key_file=key_conll_path,
            sys_file=response_conll_path,
            NP_only=False,
            remove_nested=False,
            keep_singletons=True,
            min_span=False
        )

        # Define metrics to compute (same as training/validation)
        metrics_to_compute = [
            ('mentions', evaluator.mentions),
            ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed),
            ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)
        ]

        # Compute each metric
        # Note: This project uses a custom coval that requires dict_phrasing_diversity, dataset_name, mention_type
        # For metrics other than pd_LEA, we pass an empty dict for phrasing diversity
        dict_phrasing_diversity = {}

        results = []
        for metric_name, metric_fn in metrics_to_compute:
            recall, precision, f1 = evaluator.evaluate_documents(
                doc_coref_infos,
                dict_phrasing_diversity,
                metric_fn,
                dataset_name=dataset_name,
                mention_type='all',
                beta=1
            )

            results.append({
                'metric': metric_name,
                'Recall': recall * 100,
                'Precision': precision * 100,
                'F1': f1 * 100,
                'training_config': training_config,
                'dataset': dataset_name,
                'mode': mode,
                'num_mentions': num_mentions
            })

        # Compute CoNLL score (average of MUC, B³, CEAF)
        muc_f1 = [r for r in results if r['metric'] == 'muc'][0]['F1']
        bcub_f1 = [r for r in results if r['metric'] == 'bcub'][0]['F1']
        ceafe_f1 = [r for r in results if r['metric'] == 'ceafe'][0]['F1']
        conll_f1 = (muc_f1 + bcub_f1 + ceafe_f1) / 3

        # Compute CoNLL precision and recall similarly
        muc_p = [r for r in results if r['metric'] == 'muc'][0]['Precision']
        bcub_p = [r for r in results if r['metric'] == 'bcub'][0]['Precision']
        ceafe_p = [r for r in results if r['metric'] == 'ceafe'][0]['Precision']
        conll_p = (muc_p + bcub_p + ceafe_p) / 3

        muc_r = [r for r in results if r['metric'] == 'muc'][0]['Recall']
        bcub_r = [r for r in results if r['metric'] == 'bcub'][0]['Recall']
        ceafe_r = [r for r in results if r['metric'] == 'ceafe'][0]['Recall']
        conll_r = (muc_r + bcub_r + ceafe_r) / 3

        results.append({
            'metric': 'conll',
            'Recall': conll_r,
            'Precision': conll_p,
            'F1': conll_f1,
            'training_config': training_config,
            'dataset': dataset_name,
            'mode': mode,
            'num_mentions': num_mentions
        })

        df_metrics = pd.DataFrame(results)
        return df_metrics

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_all_inference_results(
    mentions_df: pd.DataFrame,
    output_base_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate all inference results in the output directory.

    Args:
        mentions_df: Mentions DataFrame with metadata
        output_base_dir: Base directory containing inference outputs

    Returns:
        Dictionary mapping "metrics_summary" to consolidated metrics DataFrame
    """
    logger.info("="*80)
    logger.info("EVALUATION PIPELINE")
    logger.info("="*80)

    output_path = Path(output_base_dir)
    all_results = []

    # Find all training configurations
    training_configs = [d.name for d in output_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(training_configs)} training configurations")

    for training_config in sorted(training_configs):
        config_dir = output_path / training_config
        logger.info(f"\nEvaluating configuration: {training_config}")

        # Find all datasets
        datasets = [d.name for d in config_dir.iterdir() if d.is_dir()]

        for dataset_name in sorted(datasets):
            dataset_dir = config_dir / dataset_name

            # Find all modes
            modes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]

            for mode in sorted(modes):
                mode_dir = dataset_dir / mode

                # Check for key.conll and response.conll
                key_path = mode_dir / 'key.conll'
                response_path = mode_dir / 'response.conll'

                if not key_path.exists() or not response_path.exists():
                    logger.warning(f"Missing CoNLL files in {mode_dir}")
                    continue

                # Evaluate this configuration
                metrics_df = evaluate_single_configuration(
                    mentions_df=mentions_df,
                    training_config=training_config,
                    dataset_name=dataset_name,
                    mode=mode,
                    key_conll_path=str(key_path),
                    response_conll_path=str(response_path)
                )

                if metrics_df is not None:
                    all_results.append(metrics_df)

    # Consolidate all results
    if all_results:
        # Combine all metrics
        consolidated_df = pd.concat(all_results, ignore_index=True)

        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total evaluations: {len(all_results)}")
        logger.info(f"Training configs: {len(training_configs)}")

        # Print summary statistics - filter to CoNLL metric only
        conll_df = consolidated_df[consolidated_df['metric'] == 'conll']
        if len(conll_df) > 0:
            logger.info(f"\nCoNLL F1 Summary:")
            summary = conll_df.groupby('training_config')['F1'].agg(['mean', 'std', 'min', 'max'])
            logger.info(f"\n{summary}")

        return {"metrics_summary": consolidated_df}
    else:
        logger.warning("No evaluation results generated")
        return {"metrics_summary": pd.DataFrame()}
