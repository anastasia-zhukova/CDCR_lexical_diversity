"""
Shared model and evaluation utilities used by both train.py and infer_news.py.

Keeping these here avoids circular imports when infer_news.py is run in a
Databricks session that already has train.py partially initialized.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from .metrics import compute_metrics_from_assignments


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CorefEncoder(nn.Module):
    """
    Coreference mention encoder: RoBERTa-large + dual MLP heads.

    Produces a 2048-dim embedding per mention by concatenating:
      - CLS token embedding projected through MLP (1024-dim)
      - Mention span embedding (sum of [E]...[/E] tokens) projected through MLP (1024-dim)
    """

    def __init__(self, tokenizer, mlp_hidden_dim=1024):
        super().__init__()
        self.language_model = RobertaModel.from_pretrained("roberta-large")
        self.language_model.resize_token_embeddings(len(tokenizer))

        self.e_start_id = tokenizer.convert_tokens_to_ids("[E]")
        self.e_end_id = tokenizer.convert_tokens_to_ids("[/E]")
        self.mlp_hidden_dim = mlp_hidden_dim
        hidden_size = self.language_model.config.hidden_size  # 1024

        self.mlp_cls = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )
        self.mlp_mention = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

    def forward(self, input_ids, attention_mask):
        e_start_idx = input_ids.eq(self.e_start_id).nonzero(as_tuple=True)
        e_end_idx = input_ids.eq(self.e_end_id).nonzero(as_tuple=True)

        outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

        cls_embed = hidden[:, 0, :]

        seq_indices = torch.arange(hidden.size(1), device=hidden.device)
        start_mask = seq_indices >= e_start_idx[1].unsqueeze(1)
        end_mask = seq_indices <= e_end_idx[1].unsqueeze(1)
        mention_mask = start_mask & end_mask
        mention_embed = (hidden * mention_mask.unsqueeze(2).float()).sum(dim=1)

        out_cls = self.mlp_cls(cls_embed)
        out_mention = self.mlp_mention(mention_embed)
        concat = torch.cat([out_cls, out_mention], dim=1)

        return F.normalize(concat, p=2, dim=1)


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """
    Compute embeddings for all mentions in a dataloader.

    Returns:
        embeddings: (N, D) numpy array of L2-normalized embeddings.
        labels: list of string chain IDs.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        embeddings = model(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(batch["label"])

    return np.concatenate(all_embeddings, axis=0), all_labels


# ---------------------------------------------------------------------------
# Clustering evaluation (MUC, B-CUBED, CEAFe, LEA, pd_LEA)
# ---------------------------------------------------------------------------

# Public dict keys used throughout this module (and its callers) have always
# used the "_precision"/"_recall" suffix, while metrics.py uses the shorter
# "_p"/"_r" suffix. Map between the two so existing call sites don't need to
# change.
_P_R_KEY_MAP = {
    "muc_p": "muc_precision", "muc_r": "muc_recall",
    "b3_p": "b3_precision", "b3_r": "b3_recall",
    "ceafe_p": "ceafe_precision", "ceafe_r": "ceafe_recall",
    "lea_p": "lea_precision", "lea_r": "lea_recall",
    "pd_lea_p": "pd_lea_precision", "pd_lea_r": "pd_lea_recall",
}


def _compute_all_metrics(cluster_ids, gold_labels, pd_tuples):
    """Compute all coreference metrics from cluster assignments.

    Thin wrapper around :func:`metrics.compute_metrics_from_assignments`
    that renames the ``*_p``/``*_r`` keys to the ``*_precision``/``*_recall``
    keys expected by this module's existing callers.
    """
    result = compute_metrics_from_assignments(cluster_ids, gold_labels, pd_tuples=pd_tuples)
    return {_P_R_KEY_MAP.get(k, k): v for k, v in result.items()}


def evaluate_clustering(embeddings, gold_labels, pd_tuples, threshold=0.5):
    """
    Agglomerative clustering + full coreference evaluation.

    Clusters all mentions together (dataset-level).

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        gold_labels: list of string chain IDs (len N).
        pd_tuples: list of (mention_head, mention_wordset, mention_wo_stopwords).
        threshold: cosine distance threshold for clustering.

    Returns:
        dict with all metric families.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    distances = pdist(embeddings, metric="cosine")
    Z = linkage(distances, method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")

    return _compute_all_metrics(cluster_ids, gold_labels, pd_tuples)


def evaluate_per_subtopic(embeddings, gold_labels, pd_tuples, subtopic_ids,
                          threshold=0.5):
    """
    Per-subtopic agglomerative clustering + coreference evaluation.

    Clusters mentions independently within each subtopic (matching the standard
    CDCR evaluation protocol), then computes metrics across all mentions.

    Singletons within a subtopic (only 1 non-singleton mention) are assigned
    their own unique cluster.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        gold_labels: list of string chain IDs (len N).
        pd_tuples: list of (mention_head, mention_wordset, mention_wo_stopwords).
        subtopic_ids: list of subtopic identifiers (len N).
        threshold: cosine distance threshold for clustering.

    Returns:
        dict with all metric families.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(gold_labels)
    cluster_ids = np.zeros(n, dtype=np.int32)
    next_cluster = 1  # fcluster returns 1-based IDs

    # Group mention indices by subtopic
    subtopic_to_indices = defaultdict(list)
    for i, st_id in enumerate(subtopic_ids):
        subtopic_to_indices[st_id].append(i)

    for st_id, indices in subtopic_to_indices.items():
        if len(indices) < 2:
            # Single mention in subtopic — assign its own cluster
            cluster_ids[indices[0]] = next_cluster
            next_cluster += 1
            continue

        sub_embs = embeddings[indices]
        distances = pdist(sub_embs, metric="cosine")
        Z = linkage(distances, method="average")
        sub_clusters = fcluster(Z, t=threshold, criterion="distance")

        # Offset cluster IDs to make them globally unique
        for local_i, global_i in enumerate(indices):
            cluster_ids[global_i] = sub_clusters[local_i] + next_cluster
        next_cluster += sub_clusters.max() + 1

    return _compute_all_metrics(cluster_ids, gold_labels, pd_tuples)
