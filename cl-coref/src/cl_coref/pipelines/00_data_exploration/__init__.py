"""
Data Exploration Pipeline.

This pipeline loads all CDCR mention data into a unified format for analysis.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
