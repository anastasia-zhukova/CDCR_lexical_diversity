"""
Coreference resolution metrics: MUC, B³, CEAFe, LEA, pd_LEA, and CoNLL F1.

This module is the single source of truth for all coreference metric
implementations.  Both ``coref_utils.py`` (training / inference) and
``eval_checkpoint.py`` (CoNLL file generation + Perl-scaler cross-check) import
from here rather than maintaining their own copies.

Two high-level entry points are provided:

* ``compute_metrics_from_clusters(pred_clusters, gold_clusters, …)``
    Takes pre-built cluster dicts (``id → set of mention indices``).
    Used by ``eval_checkpoint.py`` where clusters are sliced from a DataFrame.

* ``compute_metrics_from_assignments(cluster_ids, gold_labels, pd_tuples)``
    Takes 1-D arrays of predicted / gold cluster assignments and builds
    the cluster dicts internally.  Optionally computes pd_LEA when
    ``pd_tuples`` is supplied.  Used by ``coref_utils.py``.

References
----------
- MUC:     Vilain et al. (1995)
- B³:      Bagga & Baldwin (1998)
- CEAFe:   Luo (2005)
- LEA:     Moosavi & Strube (2016)
- pd_LEA:  phrasing-diversity-weighted LEA (project-specific)
"""

from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def f1(p, r):
    """F1 from precision and recall."""
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Low-level metric primitives
# ---------------------------------------------------------------------------

def muc(clusters, mention_to_other):
    """MUC link-based metric (Vilain et al., 1995).

    Parameters
    ----------
    clusters : list[set]
        The clusters to score (predicted for precision, gold for recall).
    mention_to_other : dict
        Mapping ``mention → set of mentions`` in the *other* partition
        (gold for precision, predicted for recall).

    Returns
    -------
    (tp, p) : tuple[int, int]
        True-positive links and total possible links.
    """
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


def b_cubed(clusters, mention_to_other_cluster):
    """B-CUBED mention-based metric (Bagga & Baldwin, 1998).

    Returns
    -------
    (num, den) : tuple[float, float]
        Numerator and denominator for the precision/recall ratio.
    """
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


def ceafe(pred_list, gold_list):
    """CEAFe entity-based metric with phi4 similarity (Luo, 2005).

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    to find the optimal one-to-one matching between predicted and gold
    clusters.

    Returns
    -------
    (p_num, p_den, r_num, r_den) : tuple[float, float, float, float]
    """
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros((len(gold_list), len(pred_list)))
    for i, gc in enumerate(gold_list):
        for j, pc in enumerate(pred_list):
            overlap = len(gc & pc)
            scores[i, j] = 2 * overlap / (len(gc) + len(pc)) if (len(gc) + len(pc)) > 0 else 0
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(pred_list), similarity, len(gold_list)


def phrasing_diversity_calc(mentions) -> float:
    """
    Calculate phrasing diversity for a coreference chain.

    Replicates CL_COREF/coval/eval/evaluator.py::phrasing_diversity_calc.

    Args:
        mentions: list of (mention_head, mention_wordset, mention_wo_stopwords) tuples.
            mention_wordset is a comma-separated string of sorted unique content words.
            mention_wo_stopwords is a string repr of list of content words.

    Returns:
        PD score (float), or 1.0 for singleton chains.
    """
    headwords_phrase_tree = {}
    for mention_head, mention_wordset, mention_wo_stopwords in mentions:
        if not mention_wo_stopwords:
            continue
        if mention_head not in headwords_phrase_tree:
            headwords_phrase_tree[mention_head] = {
                "set": {mention_wordset},
                "list": [mention_wo_stopwords],
            }
        else:
            headwords_phrase_tree[mention_head]["set"].add(mention_wordset)
            headwords_phrase_tree[mention_head]["list"].append(mention_wo_stopwords)

    sets = []
    fractions = []
    for _, head_properties in headwords_phrase_tree.items():
        fractions.append(len(head_properties["set"]) / len(head_properties["list"]))
        sets.append(len(head_properties["set"]))

    n = len(mentions)
    score = np.sum(fractions) * np.sum(sets) / n if n > 1 else 1.0
    return float(format(score, ".3f"))


def lea(input_clusters, mention_to_output, weight_fn):
    """Link-Based Entity-Aware F1 (Moosavi & Strube, 2016).

    Parameters
    ----------
    input_clusters : list[set]
        Clusters to score (predicted for precision, gold for recall).
    mention_to_output : dict
        Mapping ``mention → set of mentions`` in the *other* partition.
    weight_fn : callable
        Function ``set → float`` that assigns a weight to each cluster.
        Standard LEA uses ``len``; pd_LEA uses a phrasing-diversity weight.

    Returns
    -------
    (num, den) : tuple[float, float]
    """
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


# ---------------------------------------------------------------------------
# High-level: from pre-built cluster dicts
# ---------------------------------------------------------------------------

def compute_metrics_from_clusters(pred_clusters, gold_clusters,
                                  pd_weight_fn=None):
    """Compute MUC, B³, CEAFe, LEA (and optionally pd_LEA) from
    pre-built cluster dictionaries.

    Parameters
    ----------
    pred_clusters : dict[int, set[int]]
        Predicted clusters: cluster_id → set of mention indices.
    gold_clusters : dict[str, set[int]]
        Gold clusters: chain_id → set of mention indices.
    pd_weight_fn : callable or None
        If provided, also compute pd_LEA using this weight function.
        If None, pd_LEA is omitted from the result.

    Returns
    -------
    dict with keys:
        muc_p, muc_r, muc_f1,
        b3_p, b3_r, b3_f1,
        ceafe_p, ceafe_r, ceafe_f1,
        conll_f1,
        lea_p, lea_r, lea_f1,
        pd_lea_p, pd_lea_r, pd_lea_f1   (only when *pd_weight_fn* is given),
        n_gold_clusters, n_pred_clusters
    """
    pred_list = list(pred_clusters.values())
    gold_list = list(gold_clusters.values())

    mention_to_gold = {m: c for c in gold_list for m in c}
    mention_to_pred = {m: c for c in pred_list for m in c}

    # --- MUC ---
    muc_p_num, muc_p_den = muc(pred_list, mention_to_gold)
    muc_r_num, muc_r_den = muc(gold_list, mention_to_pred)
    muc_p = muc_p_num / muc_p_den if muc_p_den > 0 else 0
    muc_r = muc_r_num / muc_r_den if muc_r_den > 0 else 0
    muc_f1 = f1(muc_p, muc_r)

    # --- B³ ---
    b3_p_num, b3_p_den = b_cubed(pred_list, mention_to_gold)
    b3_r_num, b3_r_den = b_cubed(gold_list, mention_to_pred)
    b3_p = b3_p_num / b3_p_den if b3_p_den > 0 else 0
    b3_r = b3_r_num / b3_r_den if b3_r_den > 0 else 0
    b3_f1 = f1(b3_p, b3_r)

    # --- CEAFe ---
    ceafe_p_num, ceafe_p_den, ceafe_r_num, ceafe_r_den = ceafe(pred_list, gold_list)
    ceafe_p = ceafe_p_num / ceafe_p_den if ceafe_p_den > 0 else 0
    ceafe_r = ceafe_r_num / ceafe_r_den if ceafe_r_den > 0 else 0
    ceafe_f1 = f1(ceafe_p, ceafe_r)

    conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3.0

    # --- LEA ---
    lea_r_num, lea_r_den = lea(gold_list, mention_to_pred, len)
    lea_p_num, lea_p_den = lea(pred_list, mention_to_gold, len)
    lea_r = lea_r_num / lea_r_den if lea_r_den > 0 else 0
    lea_p = lea_p_num / lea_p_den if lea_p_den > 0 else 0
    lea_f1 = f1(lea_p, lea_r)

    result = {
        "muc_p": muc_p, "muc_r": muc_r, "muc_f1": muc_f1,
        "b3_p": b3_p, "b3_r": b3_r, "b3_f1": b3_f1,
        "ceafe_p": ceafe_p, "ceafe_r": ceafe_r, "ceafe_f1": ceafe_f1,
        "conll_f1": conll_f1,
        "lea_p": lea_p, "lea_r": lea_r, "lea_f1": lea_f1,
        "n_gold_clusters": len(gold_list),
        "n_pred_clusters": len(pred_list),
    }

    # --- pd_LEA (optional) ---
    if pd_weight_fn is not None:
        pd_r_num, pd_r_den = lea(gold_list, mention_to_pred, pd_weight_fn)
        pd_p_num, pd_p_den = lea(pred_list, mention_to_gold, pd_weight_fn)
        pd_lea_r = pd_r_num / pd_r_den if pd_r_den > 0 else 0
        pd_lea_p = pd_p_num / pd_p_den if pd_p_den > 0 else 0
        pd_lea_f1 = f1(pd_lea_p, pd_lea_r)
        result["pd_lea_p"] = pd_lea_p
        result["pd_lea_r"] = pd_lea_r
        result["pd_lea_f1"] = pd_lea_f1

    return result


# ---------------------------------------------------------------------------
# High-level: from cluster assignment arrays
# ---------------------------------------------------------------------------

def compute_metrics_from_assignments(cluster_ids, gold_labels, pd_tuples=None):
    """Compute all coreference metrics from 1-D assignment arrays.

    Builds predicted / gold cluster dicts internally, then delegates to
    :func:`compute_metrics_from_clusters`.

    Parameters
    ----------
    cluster_ids : array-like[int]
        Predicted cluster ID for each mention (len N).
    gold_labels : list[str]
        Gold chain ID for each mention (len N).
    pd_tuples : list or None
        If provided, each element is
        ``(mention_head, mention_wordset, mention_wo_stopwords)`` and
        pd_LEA is computed using a phrasing-diversity weight.

    Returns
    -------
    dict
        Same keys as :func:`compute_metrics_from_clusters`, plus
        ``pd_lea_*`` keys when *pd_tuples* is provided.
    """
    pred_clusters = defaultdict(set)
    gold_clusters = defaultdict(set)
    for i, (pred_id, gold_id) in enumerate(zip(cluster_ids, gold_labels)):
        pred_clusters[pred_id].add(i)
        gold_clusters[gold_id].add(i)

    pd_weight_fn = None
    if pd_tuples is not None:
        def pd_weight_fn(cluster):
            if len(cluster) == 1:
                return 0.0
            mentions = [pd_tuples[i] for i in cluster if pd_tuples[i][2] is not None]
            if len(mentions) < 2:
                return 0.0
            return phrasing_diversity_calc(mentions)

    return compute_metrics_from_clusters(pred_clusters, gold_clusters,
                                         pd_weight_fn=pd_weight_fn)
