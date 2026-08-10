"""
Contrastive pair sampler — hard-mined positive/negative mention pairs for
contrastive-loss training (the alternative to MS Loss + PK batch sampling).

Ported from cl-coref-orig's data_preparation/{sampler,full_sampler}.py and
adapted to ms-coref's processed parquet schema (mention_sentence_context /
document_context / mention_wordset are already precomputed by
data_preparation.py, so no raw-context reconstruction is needed here).

Algorithm (Hsu & Horwood, NAACL 2022):
  1. Keep non-singleton mentions only.
  2. Build unique sentences from (doc_id, sent_id) and embed them with SBERT.
  3. Positives: for each (topic_id, coref_chain) cluster, build a minimum
     spanning tree where edge weight is the cosine similarity of the
     mentions' sentences (Kruskal: N-1 least-similar edges that keep the
     cluster connected) — the paper's "sequences that were least similar".
  4. Negatives: for each anchor (mention in a positive pair), keep its K
     hardest same-subtopic non-coreferent partners (top-K most-similar
     non-coreferent mentions in the same subtopic).
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Hard negatives per anchor (top-K nearest non-coreferent mentions in the
# same subtopic).
K_HARD_NEGATIVES = 20

# Mirrors conll_eval.py's event/entity split rules (dataset-specific: NIdent
# and NP4E use "EVENT" as a named-entity NER category, not event coref).
_EVENT_IS_ENTITY_DATASETS = frozenset({"NIdent", "NP4E"})


def _is_event_mask(mention_full_type: pd.Series, dataset: str) -> pd.Series:
    fts = mention_full_type.fillna("")
    is_action = fts.str.contains("ACTION", case=False, na=False)
    is_event_type = fts.eq("EVENT") & (dataset not in _EVENT_IS_ENTITY_DATASETS)
    return is_action | is_event_type


def _build_entity_type(mentions_df: pd.DataFrame) -> pd.Series:
    """Return coarse 'events'/'entities' labels, one per dataset present."""
    entity_type = pd.Series("entities", index=mentions_df.index, dtype="string")
    for dataset, idx in mentions_df.groupby("dataset").groups.items():
        sub = mentions_df.loc[idx, "mention_full_type"]
        is_ev = _is_event_mask(sub, dataset)
        entity_type.loc[is_ev.index[is_ev]] = "events"
    return entity_type


def _mention_sentence_text(mention: dict) -> str:
    """Strip the [E]/[/E] markers — SBERT wasn't trained on them."""
    sent = str(mention.get("mention_sentence_context") or "")
    return sent.replace("[E]", "").replace("[/E]", "").replace("  ", " ").strip()


# ---------------------------------------------------------------------------
# Sentence embedding
# ---------------------------------------------------------------------------


def build_sentence_index(mentions: list[dict]) -> tuple[list[str], list[int]]:
    """Deduplicate sentences and map each mention to its sentence index."""
    sent_key_to_idx: dict[tuple[Any, Any], int] = {}
    unique_sentences: list[str] = []

    for m in mentions:
        key = (m["doc_id"], m["sent_id"])
        if key not in sent_key_to_idx:
            sent_key_to_idx[key] = len(unique_sentences)
            unique_sentences.append(_mention_sentence_text(m))

    mention_to_sent = [sent_key_to_idx[(m["doc_id"], m["sent_id"])] for m in mentions]
    return unique_sentences, mention_to_sent


def compute_sentence_similarity(
    sentences: list[str], model: "SentenceTransformer"
) -> np.ndarray:
    """Embed unique sentences and return their cosine similarity matrix."""
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings @ embeddings.T


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------


def kruskal_mst_positives(
    cluster_idxs: list[int],
    mention_to_sent: list[int],
    sent_sim: np.ndarray,
) -> list[tuple[int, int, float]]:
    """
    Return the N-1 least-similar edges that span the cluster (Kruskal MST).

    Edge weight = cosine similarity of the mentions' sentences. Edges are
    sorted by similarity ascending and added via union-find as long as they
    don't form a cycle, until the cluster is connected.
    """
    if len(cluster_idxs) < 2:
        return []

    edges: list[tuple[float, int, int]] = []
    for a, b in itertools.combinations(cluster_idxs, 2):
        sim = float(sent_sim[mention_to_sent[a], mention_to_sent[b]])
        edges.append((sim, a, b))
    edges.sort()  # ascending similarity -> least similar first

    parent = {i: i for i in cluster_idxs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    pairs: list[tuple[int, int, float]] = []
    target = len(cluster_idxs) - 1
    for sim, a, b in edges:
        if union(a, b):
            pairs.append((a, b, sim))
            if len(pairs) == target:
                break
    return pairs


def build_knn_same_subtopic_negatives(
    mentions: list[dict],
    mention_to_sent: list[int],
    sent_sim: np.ndarray,
    anchors: set[int],
    k: int,
) -> list[tuple[int, int, float]]:
    """
    For every anchor, keep its K most-similar non-coreferent partners that
    live in the same subtopic. Returns deduplicated (i, j, sim) pairs with
    i < j.
    """
    subtopic_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, m in enumerate(mentions):
        subtopic_to_indices[m["subtopic_id"]].append(idx)

    selected: set[tuple[int, int]] = set()
    pair_sim: dict[tuple[int, int], float] = {}

    for anchor in anchors:
        anchor_sub = mentions[anchor]["subtopic_id"]
        anchor_chain = mentions[anchor]["coref_chain"]

        candidates: list[tuple[float, int]] = []
        for other in subtopic_to_indices[anchor_sub]:
            if other == anchor:
                continue
            if mentions[other]["coref_chain"] == anchor_chain:
                continue
            sim = float(sent_sim[mention_to_sent[anchor], mention_to_sent[other]])
            candidates.append((sim, other))

        candidates.sort(reverse=True)  # most-similar first
        for sim, other in candidates[:k]:
            a, b = (anchor, other) if anchor < other else (other, anchor)
            if (a, b) not in selected:
                selected.add((a, b))
                pair_sim[(a, b)] = sim

    return [(a, b, pair_sim[(a, b)]) for a, b in selected]


def build_pairs(
    mentions: list[dict],
    mention_to_sent: list[int],
    sent_sim: np.ndarray,
    k_neg: int = K_HARD_NEGATIVES,
) -> dict:
    """
    Build contrastive pairs:
      - Positives: Kruskal MST over each (topic_id, coref_chain) cluster.
      - Negatives: for each anchor in a positive pair, its K hardest
        non-coreferent partners from the same subtopic.
    """
    chains: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for idx, m in enumerate(mentions):
        chains[(m["topic_id"], m["coref_chain"])].append(idx)

    positives: list[tuple[int, int, float]] = []
    for cluster_idxs in chains.values():
        positives.extend(kruskal_mst_positives(cluster_idxs, mention_to_sent, sent_sim))

    if not positives:
        raise ValueError("No positive pairs found — check mention data.")

    anchors: set[int] = set()
    for i, j, _ in positives:
        anchors.add(i)
        anchors.add(j)

    negatives = build_knn_same_subtopic_negatives(
        mentions, mention_to_sent, sent_sim, anchors, k_neg
    )

    pairs = [(i, j, 1) for i, j, _ in positives] + [(i, j, 0) for i, j, _ in negatives]

    pos_sims = np.array([s for _, _, s in positives])
    neg_sims = np.array([s for _, _, s in negatives]) if negatives else np.array([])
    n_pos, n_neg = len(positives), len(negatives)
    n_total = n_pos + n_neg

    stats = {
        "n_pairs": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fraction_positive": round(n_pos / n_total, 2) if n_total else 0,
        "median_pos_similarity": (
            round(float(np.median(pos_sims)), 2) if n_pos else None
        ),
        "median_neg_similarity": (
            round(float(np.median(neg_sims)), 2) if n_neg else None
        ),
        "k_hard_negatives": k_neg,
    }

    return {"pairs": pairs, "stats": stats}


# ---------------------------------------------------------------------------
# Kedro/pipeline-facing entry point
# ---------------------------------------------------------------------------


def run_one(
    mentions: list[dict[str, Any]],
    model: "SentenceTransformer",
    k_neg: int,
    group_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the sampler on a single grouped slice, return pair rows + stats."""
    mentions = [m for m in mentions if not m.get("is_singleton")]
    if len(mentions) < 2:
        return [], {
            "n_non_singleton_mentions": len(mentions),
            "n_pairs": 0,
            "skipped": "not enough non-singleton mentions",
        }

    unique_sents, mention_to_sent = build_sentence_index(mentions)
    sent_sim = compute_sentence_similarity(unique_sents, model)

    try:
        result = build_pairs(mentions, mention_to_sent, sent_sim, k_neg=k_neg)
    except ValueError as exc:
        return [], {
            "n_non_singleton_mentions": len(mentions),
            "n_pairs": 0,
            "skipped": str(exc),
        }

    rows: list[dict[str, Any]] = []
    for i, j, label in result["pairs"]:
        m_i, m_j = mentions[i], mentions[j]
        rows.append(
            {
                "id_mention1": m_i["mention_id"],
                "id_mention2": m_j["mention_id"],
                "coref_chain_mention1": m_i["coref_chain"],
                "coref_chain_mention2": m_j["coref_chain"],
                "coreference": bool(label),
                **group_meta,
            }
        )

    stats = dict(result["stats"])
    stats["n_non_singleton_mentions"] = len(mentions)
    stats["n_unique_sentences"] = len(unique_sents)
    return rows, stats


def sample_pairs_from_mentions(
    mention_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    Sample contrastive mention pairs from the ms-coref processed mentions table.

    Params:
      - split: split to sample from (default: "train")
      - group_by: independent-sampling-slice columns (default: ["dataset"]);
                  "entity_type" is always appended
      - sbert_model: sentence embedding model name
      - k_hard_negatives: K hardest negatives per anchor
      - max_groups: optional cap on number of groups (smoke runs)
      - max_mentions_per_group: optional cap on mentions per group
      - random_seed: seed for capped group sampling

    Returns:
        DataFrame with id_mention1, id_mention2, coref_chain_mention1/2,
        coreference (bool), entity_type, + group_by columns.
    """
    params = params or {}
    split = params.get("split", "train")
    group_by = list(params.get("group_by", ["dataset"])) or ["dataset"]
    model_name = params.get("sbert_model", SBERT_MODEL)
    k_neg = int(params.get("k_hard_negatives", K_HARD_NEGATIVES))
    max_groups = params.get("max_groups")
    max_mentions_per_group = params.get("max_mentions_per_group")
    random_seed = int(params.get("random_seed", 13))

    working_df = mention_df.copy()
    working_df["entity_type"] = _build_entity_type(working_df)
    if "entity_type" not in group_by:
        group_by = [*group_by, "entity_type"]

    if split and "split" in working_df.columns:
        normalized = (
            working_df["split"]
            .astype("string")
            .str.lower()
            .replace({"validation": "val"})
        )
        split_norm = str(split).lower().replace("validation", "val")
        working_df = working_df[normalized == split_norm].copy()

    empty_cols = [
        "id_mention1",
        "id_mention2",
        "coref_chain_mention1",
        "coref_chain_mention2",
        "coreference",
        *group_by,
    ]
    if working_df.empty:
        return pd.DataFrame(columns=empty_cols)

    logger.info("Loading SBERT model for pair sampling: %s", model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    grouped = list(working_df.groupby(group_by, dropna=False))
    if max_groups is not None:
        grouped = grouped[: int(max_groups)]

    all_rows: list[dict[str, Any]] = []
    for key, group_df in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_meta = dict(zip(group_by, key))

        if max_mentions_per_group is not None and len(group_df) > int(
            max_mentions_per_group
        ):
            group_df = group_df.sample(
                n=int(max_mentions_per_group), random_state=random_seed
            )

        rows, stats = run_one(
            mentions=group_df.to_dict("records"),
            model=model,
            k_neg=k_neg,
            group_meta=group_meta,
        )
        logger.info("  %s: %s", group_meta, stats)
        all_rows.extend(rows)

    pairs_df = pd.DataFrame(all_rows, columns=empty_cols)
    if not pairs_df.empty:
        pairs_df["coreference"] = pairs_df["coreference"].astype(bool)
    return pairs_df
