"""
Pipeline for pairs selection using sentence-level similarities.

This pipeline creates mention pairs based on sentence-level SBERT similarities,
following the strategy from the original sampling.py implementation.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
