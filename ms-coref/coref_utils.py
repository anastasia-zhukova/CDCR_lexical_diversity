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

from data_preparation import phrasing_diversity_calc


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

def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _compute_muc(clusters, mention_to_other):
    """MUC link-based metric (Vilain et al., 1995)."""
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_other:
                linked.add(id(mention_to_other[m]))
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def _compute_b_cubed(clusters, mention_to_other_cluster):
    """B-CUBED mention-based metric (Bagga & Baldwin, 1998)."""
    from collections import Counter
    num, den = 0.0, 0.0
    for c in clusters:
        gold_counts = Counter()
        for m in c:
            if m in mention_to_other_cluster:
                gold_counts[id(mention_to_other_cluster[m])] += 1
        correct = sum(cnt * cnt for cnt in gold_counts.values())
        num += correct / len(c)
        den += len(c)
    return num, den


def _compute_ceafe(pred_list, gold_list):
    """CEAFe entity-based metric with phi4 similarity (Luo, 2005)."""
    from scipy.optimize import linear_sum_assignment
    scores = np.zeros((len(gold_list), len(pred_list)))
    for i, gc in enumerate(gold_list):
        for j, pc in enumerate(pred_list):
            overlap = len(gc & pc)
            scores[i, j] = 2 * overlap / (len(gc) + len(pc)) if (len(gc) + len(pc)) > 0 else 0
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(pred_list), similarity, len(gold_list)


def _compute_lea(input_clusters, mention_to_output, weight_fn):
    num, den = 0.0, 0.0
    for cluster in input_clusters:
        w = weight_fn(cluster)
        if len(cluster) == 1:
            m = next(iter(cluster))
            den += w
            if len(mention_to_output.get(m, set())) == 1:
                num += w
        else:
            den += w
            all_links = len(cluster) * (len(cluster) - 1) / 2.0
            common_links = 0.0
            for m in cluster:
                out_cluster = mention_to_output.get(m, set())
                common_links += len(cluster & out_cluster) - 1
            common_links /= 2.0
            num += w * (common_links / all_links) if all_links > 0 else 0.0
    return num, den


def _compute_all_metrics(cluster_ids, gold_labels, pd_tuples):
    """Compute all coreference metrics from cluster assignments."""
    pred_clusters = defaultdict(set)
    gold_clusters = defaultdict(set)
    for i, (pred_id, gold_id) in enumerate(zip(cluster_ids, gold_labels)):
        pred_clusters[pred_id].add(i)
        gold_clusters[gold_id].add(i)

    pred_list = list(pred_clusters.values())
    gold_list = list(gold_clusters.values())

    mention_to_gold = {m: mentions for mentions in gold_clusters.values() for m in mentions}
    mention_to_pred = {m: mentions for mentions in pred_clusters.values() for m in mentions}

    # --- MUC ---
    muc_p_num, muc_p_den = _compute_muc(pred_list, mention_to_gold)
    muc_r_num, muc_r_den = _compute_muc(gold_list, mention_to_pred)
    muc_p = muc_p_num / muc_p_den if muc_p_den > 0 else 0
    muc_r = muc_r_num / muc_r_den if muc_r_den > 0 else 0
    muc_f1 = _f1(muc_p, muc_r)

    # --- B-CUBED ---
    b3_p_num, b3_p_den = _compute_b_cubed(pred_list, mention_to_gold)
    b3_r_num, b3_r_den = _compute_b_cubed(gold_list, mention_to_pred)
    b3_p = b3_p_num / b3_p_den if b3_p_den > 0 else 0
    b3_r = b3_r_num / b3_r_den if b3_r_den > 0 else 0
    b3_f1 = _f1(b3_p, b3_r)

    # --- CEAFe ---
    ceafe_p_num, ceafe_p_den, ceafe_r_num, ceafe_r_den = _compute_ceafe(pred_list, gold_list)
    ceafe_p = ceafe_p_num / ceafe_p_den if ceafe_p_den > 0 else 0
    ceafe_r = ceafe_r_num / ceafe_r_den if ceafe_r_den > 0 else 0
    ceafe_f1 = _f1(ceafe_p, ceafe_r)

    conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3.0

    # --- LEA ---
    lea_r_num, lea_r_den = _compute_lea(gold_list, mention_to_pred, len)
    lea_p_num, lea_p_den = _compute_lea(pred_list, mention_to_gold, len)
    lea_r = lea_r_num / lea_r_den if lea_r_den > 0 else 0
    lea_p = lea_p_num / lea_p_den if lea_p_den > 0 else 0
    lea_f1 = _f1(lea_p, lea_r)

    # --- pd_LEA ---
    def pd_weight(cluster):
        if len(cluster) == 1:
            return 0.0
        mentions = [pd_tuples[i] for i in cluster if pd_tuples[i][2] is not None]
        if len(mentions) < 2:
            return 0.0
        return phrasing_diversity_calc(mentions)

    pd_r_num, pd_r_den = _compute_lea(gold_list, mention_to_pred, pd_weight)
    pd_p_num, pd_p_den = _compute_lea(pred_list, mention_to_gold, pd_weight)
    pd_lea_r = pd_r_num / pd_r_den if pd_r_den > 0 else 0
    pd_lea_p = pd_p_num / pd_p_den if pd_p_den > 0 else 0
    pd_lea_f1 = _f1(pd_lea_p, pd_lea_r)

    return {
        "muc_precision": muc_p, "muc_recall": muc_r, "muc_f1": muc_f1,
        "b3_precision": b3_p, "b3_recall": b3_r, "b3_f1": b3_f1,
        "ceafe_precision": ceafe_p, "ceafe_recall": ceafe_r, "ceafe_f1": ceafe_f1,
        "conll_f1": conll_f1,
        "lea_precision": lea_p, "lea_recall": lea_r, "lea_f1": lea_f1,
        "pd_lea_precision": pd_lea_p, "pd_lea_recall": pd_lea_r, "pd_lea_f1": pd_lea_f1,
        "n_gold_clusters": len(gold_list), "n_pred_clusters": len(pred_list),
    }


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
