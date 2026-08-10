"""
Shared coreference scoring primitives (MUC, B-CUBED, CEAFe, LEA).

Completely model/architecture agnostic: these operate purely on cluster
assignments (sets of mention indices), so they are reused by every training
regime (MS Loss / PK sampler, contrastive loss / pair sampler, etc.) and by
every evaluation entry point (coref_encoder.py, conll_eval.py).
"""

import numpy as np


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_muc(clusters, mention_to_other):
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


def compute_b_cubed(clusters, mention_to_other_cluster):
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


def compute_ceafe(pred_list, gold_list):
    """CEAFe entity-based metric with phi4 similarity (Luo, 2005)."""
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros((len(gold_list), len(pred_list)))
    for i, gc in enumerate(gold_list):
        for j, pc in enumerate(pred_list):
            overlap = len(gc & pc)
            scores[i, j] = (
                2 * overlap / (len(gc) + len(pc)) if (len(gc) + len(pc)) > 0 else 0
            )
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(pred_list), similarity, len(gold_list)


def compute_lea(input_clusters, mention_to_output, weight_fn):
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
