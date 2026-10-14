"""
Pipeline 05: Evaluation

This pipeline evaluates inference results by:
1. Reading CoNLL files from inference output
2. Computing comprehensive coreference metrics
3. Generating summary tables

The pipeline processes all training configurations, datasets, and modes.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_all_inference_results


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the evaluation pipeline.

    This pipeline:
    1. Loads mentions with metadata (mention_head, mention_wordset, etc.)
    2. Parses CoNLL files from all inference configurations
    3. Computes metrics: MUC, B³, CEAF-e, LEA, pd_LEA, CoNLL
    4. Generates consolidated metrics summary

    Returns:
        Kedro Pipeline for evaluation
    """
    return pipeline(
        [
            node(
                func=evaluate_all_inference_results,
                inputs=[
                    "mentions_for_pairs",  # From catalog_01
                    "params:inference_config.output_base_dir"
                ],
                outputs="metrics_summary",
                name="evaluate_all_results_node",
                tags=["evaluation"],
            ),
        ],
        tags="05_evaluation",
    )
