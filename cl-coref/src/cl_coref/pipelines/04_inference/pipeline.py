"""
Pipeline 04: Inference on Test Data

This pipeline performs test-time inference using trained models.
It evaluates models at the subtopic level and generates CoNLL files for scoring.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_full_inference_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the inference pipeline.

    This pipeline:
    1. Loads test mentions with representations
    2. Runs inference using trained model (subtopic-level clustering)
    3. Writes CoNLL files for evaluation (key.conll and response.conll)

    The pipeline processes three modes per dataset:
    - Events: Only event mentions
    - Entities: Only entity mentions
    - Mix: Both events and entities combined

    Returns:
        Kedro Pipeline for test inference
    """
    return pipeline(
        [
            node(
                func=run_full_inference_pipeline,
                inputs=[
                    "mentions_for_pairs",  # From catalog_01
                    "mention_representations",  # From catalog_01
                    "params:training_config",
                    "params:model_checkpoint_path",
                    "params:inference_config.similarity_threshold",
                    "params:inference_config.output_base_dir"
                ],
                outputs="inference_results",
                name="run_inference_node",
                tags=["inference"],
            ),
        ],
        tags="04_inference",
    )
