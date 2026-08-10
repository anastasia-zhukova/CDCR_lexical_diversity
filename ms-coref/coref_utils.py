"""
Backward-compatible re-export shim.

The model, embedding, and evaluation code that used to live in this module
has moved to coref_encoder.py (architecture + embeddings + clustering
evaluation) and metrics.py (shared MUC/B-CUBED/CEAFe/LEA primitives), so it
can be reused by both the MS Loss and contrastive-loss training regimes (see
coref_training_core.py). Existing imports of `coref_utils` keep working
unchanged.
"""

from coref_encoder import (
    CorefEncoder,
    compute_all_metrics,
    compute_embeddings,
    evaluate_clustering,
    evaluate_per_subtopic,
)

__all__ = [
    "CorefEncoder",
    "compute_all_metrics",
    "compute_embeddings",
    "evaluate_clustering",
    "evaluate_per_subtopic",
]
