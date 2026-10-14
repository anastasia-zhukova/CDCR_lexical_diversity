"""
Pipeline 04: Inference on Test Data

This pipeline performs inference on test data using trained models.
It evaluates models trained in different configurations (single, all_but, mix)
on test splits at the subtopic level.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
