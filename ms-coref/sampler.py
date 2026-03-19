"""
PK Batch Sampler for MS Loss coreference training.

Produces batches with structured negative difficulty:
  - 60% chains from same subtopic (hard negatives: shared lexical context)
  - 20% chains from same topic, different subtopic (medium negatives)
  - 20% chains from anywhere (easy negatives: diversity)
  - Remaining slots filled with singletons (weighted by token count)

Chain sampling is complexity-weighted to mimic NewsWCL50's high-PD distribution:
  weight = PD^tau * log(chain_size) * difficulty_ema^gamma

Mentions are tokenized on-the-fly using the custom tokenizer from cl-coref
that preserves [E]/[/E] markers within the max_length window.
"""

import ast
import logging
import math
import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


def parse_list(val):
    if isinstance(val, list):
        return val
    return ast.literal_eval(val)


# ---------------------------------------------------------------------------
# MentionDataset — wraps processed parquet, tokenizes on the fly
# ---------------------------------------------------------------------------

class MentionDataset(Dataset):
    """
    Dataset of coreference mentions with on-the-fly tokenization.

    Each item returns tokenized input_ids + attention_mask, plus the
    coref_chain label for MS Loss.
    """

    def __init__(self, df, tokenizer, max_length=128):
        """
        Args:
            df: DataFrame from ucdcr_mentions_processed.parquet.
            tokenizer: HuggingFace tokenizer with [E]/[/E] special tokens.
            max_length: Max token length for the custom tokenizer.
        """
        self.mention_sentence_context = df["mention_sentence_context"].tolist()
        self.document_context = df["document_context"].tolist()
        self.coref_chain = df["coref_chain"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Token IDs for [E]/[/E] — used for validation
        self.e_start_id = tokenizer.convert_tokens_to_ids("[E]")
        self.e_end_id = tokenizer.convert_tokens_to_ids("[/E]")

    def __len__(self):
        return len(self.coref_chain)

    def __getitem__(self, idx):
        result = custom_tokenize_with_special_tokens(
            self.tokenizer,
            doc_context=self.document_context[idx],
            mention_context=self.mention_sentence_context[idx],
            max_length=self.max_length,
        )
        return {
            "input_ids": torch.tensor(result["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(result["attention_mask"], dtype=torch.long),
            "label": self.coref_chain[idx],
        }


# ---------------------------------------------------------------------------
# Chain index — precomputed lookups for fast sampling
# ---------------------------------------------------------------------------

class ChainIndex:
    """
    Precomputed index over training mentions for fast PK sampling.

    Organizes mentions by chain, subtopic, and topic, and computes
    per-chain sampling weights based on PD and chain size.
    """

    def __init__(self, df, tau=1.0, gamma=0.5):
        """
        Args:
            df: Training DataFrame (already filtered to train split, singletons separated).
            tau: Exponent for PD in chain weight (higher = more PD emphasis).
            gamma: Exponent for difficulty EMA in chain weight.
        """
        self.tau = tau
        self.gamma = gamma

        # --- Build chain-level lookups ---
        self.chain_to_indices = defaultdict(list)  # chain_id -> [row indices]
        self.chain_to_subtopic = {}
        self.chain_to_topic = {}
        self.chain_to_dataset = {}
        self.chain_pd = {}
        self.chain_size = {}

        for i, row in df.iterrows():
            cid = row["coref_chain"]
            self.chain_to_indices[cid].append(i)
            if cid not in self.chain_to_subtopic:
                self.chain_to_subtopic[cid] = row["subtopic"]
                self.chain_to_topic[cid] = row["topic"]
                self.chain_to_dataset[cid] = row["dataset"]

        self.all_chains = list(self.chain_to_indices.keys())

        # Chain sizes (from index, not from df column — accounts for train-only filtering)
        for cid in self.all_chains:
            self.chain_size[cid] = len(self.chain_to_indices[cid])

        # PD scores — compute from the df columns
        self._compute_pd_scores(df)

        # --- Subtopic and topic groupings ---
        # (dataset, subtopic) -> [chain_ids]
        self.subtopic_to_chains = defaultdict(list)
        # (dataset, topic) -> [chain_ids]
        self.topic_to_chains = defaultdict(list)

        for cid in self.all_chains:
            ds = self.chain_to_dataset[cid]
            sub = self.chain_to_subtopic[cid]
            top = self.chain_to_topic[cid]
            self.subtopic_to_chains[(ds, sub)].append(cid)
            self.topic_to_chains[(ds, top)].append(cid)

        self.all_subtopic_keys = list(self.subtopic_to_chains.keys())
        self.all_topic_keys = list(self.topic_to_chains.keys())

        # --- Difficulty EMA (initialized to 1.0 for all chains) ---
        self.difficulty_ema = {cid: 1.0 for cid in self.all_chains}

        # --- Precompute static weights (PD^tau * log(size)) ---
        self._compute_static_weights()

        logger.info(
            f"ChainIndex: {len(self.all_chains)} chains, "
            f"{len(self.all_subtopic_keys)} subtopics, "
            f"{len(self.all_topic_keys)} topics"
        )

    def _compute_pd_scores(self, df):
        """Compute PD per chain from mention-level PD components."""
        # Build lookup dicts keyed by train_df index for fast access
        head_map = df["mention_head"].to_dict()
        wordset_map = df["mention_wordset"].to_dict()
        wo_stop_map = df["mention_wo_stopwords"].to_dict()

        for cid in self.all_chains:
            indices = self.chain_to_indices[cid]
            mentions = []
            for idx in indices:
                ws = wordset_map.get(idx)
                wos = wo_stop_map.get(idx)
                if ws is not None and wos is not None and isinstance(ws, str):
                    mentions.append((head_map[idx], ws, wos))

            if len(mentions) > 1:
                headwords_phrase_tree = {}
                for mh, mws, mwos in mentions:
                    if not mwos:
                        continue
                    if mh not in headwords_phrase_tree:
                        headwords_phrase_tree[mh] = {"set": {mws}, "list": [mwos]}
                    else:
                        headwords_phrase_tree[mh]["set"].add(mws)
                        headwords_phrase_tree[mh]["list"].append(mwos)
                fracs = [
                    len(v["set"]) / len(v["list"])
                    for v in headwords_phrase_tree.values()
                ]
                sets = [len(v["set"]) for v in headwords_phrase_tree.values()]
                self.chain_pd[cid] = np.sum(fracs) * np.sum(sets) / len(mentions)
            else:
                self.chain_pd[cid] = 1.0

    def _compute_static_weights(self):
        """Compute PD^tau * log(chain_size) / log(dataset_chain_count) per chain.

        The log(dataset_chain_count) damping prevents datasets with many chains
        (e.g. HyperCoref with 5,437) from dominating the sampling distribution.
        """
        # Count chains per dataset
        dataset_chain_count = Counter()
        for cid in self.all_chains:
            dataset_chain_count[self.chain_to_dataset[cid]] += 1

        self.dataset_chain_count = dict(dataset_chain_count)
        self.static_weight = {}
        for cid in self.all_chains:
            pd_val = max(self.chain_pd.get(cid, 1.0), 0.01)
            size = self.chain_size[cid]
            ds_count = self.dataset_chain_count[self.chain_to_dataset[cid]]
            self.static_weight[cid] = (
                (pd_val ** self.tau)
                * math.log(max(size, 2))
                / math.log(max(ds_count, 2))
            )

    def get_chain_weights(self, chain_ids):
        """
        Get normalized sampling weights for a list of chains.

        weight = static_weight * difficulty_ema^gamma
        """
        weights = np.array([
            self.static_weight[cid] * (self.difficulty_ema[cid] ** self.gamma)
            for cid in chain_ids
        ])
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(len(chain_ids)) / len(chain_ids)
        return weights

    def update_difficulty(self, chain_losses: dict, alpha=0.1):
        """
        Update difficulty EMA for chains based on recent loss values.

        Args:
            chain_losses: dict of chain_id -> average MS Loss contribution.
            alpha: EMA smoothing factor (higher = more responsive).
        """
        for cid, loss in chain_losses.items():
            if cid in self.difficulty_ema:
                self.difficulty_ema[cid] = (
                    (1 - alpha) * self.difficulty_ema[cid] + alpha * loss
                )


# ---------------------------------------------------------------------------
# Singleton pool — for filling batch remainders
# ---------------------------------------------------------------------------

class SingletonPool:
    """
    Pool of singleton mentions for filling batch remainders.

    Sampling is weighted by token count (longer singletons are more
    informative as negatives).
    """

    def __init__(self, df):
        """
        Args:
            df: DataFrame of singleton mentions only.
        """
        self.indices = df.index.tolist()
        if not self.indices:
            self.weights = np.array([])
            return

        # Weight by token count in mention_sentence_context
        token_counts = df["mention_sentence_context"].str.split().str.len().values
        token_counts = np.maximum(token_counts, 1).astype(float)
        self.weights = token_counts / token_counts.sum()

    def sample(self, n):
        """Sample n singleton indices, weighted by token count."""
        if not self.indices or n <= 0:
            return []
        n = min(n, len(self.indices))
        chosen = np.random.choice(
            self.indices, size=n, replace=False, p=self.weights
        )
        return chosen.tolist()


# ---------------------------------------------------------------------------
# PKBatchSampler — the core batch sampling logic
# ---------------------------------------------------------------------------

class PKBatchSampler(Sampler):
    """
    PK Batch Sampler with structured negative difficulty.

    Each batch samples P chains and takes ALL mentions from each chain,
    so every mention in a chain sees all its coreferent mentions as positives.
    Chains must have >= 2 mentions (guaranteed by build_dataloader filter).

    Batch contents:
      - 60% of chains from a chosen subtopic (hard negatives)
      - 20% of chains from the same topic, different subtopics (medium negatives)
      - 20% of chains from anywhere (easy negatives)
      - Fixed number of singletons appended

    P (chains per batch) is derived from batch_size and average chain size,
    so the expected total mentions stays close to batch_size.
    Actual batch size varies per batch since chain sizes differ.
    """

    def __init__(
        self,
        chain_index: ChainIndex,
        singleton_pool: SingletonPool,
        batch_size: int = 2048,
        subtopic_frac: float = 0.60,
        topic_frac: float = 0.20,
        random_frac: float = 0.20,
        singleton_frac: float = 0.20,
        num_batches: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            chain_index: Precomputed chain index.
            singleton_pool: Pool of singleton mentions.
            batch_size: Target total mentions per batch (used to derive P).
            subtopic_frac: Fraction of chain slots for same-subtopic chains.
            topic_frac: Fraction of chain slots for same-topic chains.
            random_frac: Fraction of chain slots for random chains.
            singleton_frac: Fraction of batch_size reserved for singletons.
            num_batches: Fixed number of batches per epoch (None = auto).
            seed: Random seed for reproducibility.
        """
        self.chain_index = chain_index
        self.singleton_pool = singleton_pool
        self.batch_size = batch_size
        self.subtopic_frac = subtopic_frac
        self.topic_frac = topic_frac
        self.random_frac = random_frac
        self.singleton_frac = singleton_frac
        self.seed = seed

        # Derive P from average chain size so expected batch ≈ batch_size
        total_chain_mentions = sum(len(v) for v in chain_index.chain_to_indices.values())
        avg_chain_size = total_chain_mentions / max(len(chain_index.all_chains), 1)
        n_pk_slots = int(batch_size * (1 - singleton_frac))
        self.n_chains = max(1, round(n_pk_slots / avg_chain_size))
        self.n_singletons = int(batch_size * singleton_frac)

        self.n_subtopic = int(self.n_chains * subtopic_frac)
        self.n_topic = int(self.n_chains * topic_frac)
        self.n_random = self.n_chains - self.n_subtopic - self.n_topic

        # Number of batches per epoch
        if num_batches is not None:
            self._num_batches = num_batches
        else:
            self._num_batches = max(1, total_chain_mentions // n_pk_slots)

        logger.info(
            f"PKBatchSampler: target_batch={batch_size}, avg_chain_size={avg_chain_size:.1f}, "
            f"chains/batch={self.n_chains} "
            f"(subtopic={self.n_subtopic}, topic={self.n_topic}, random={self.n_random}), "
            f"singletons={self.n_singletons}, batches/epoch={self._num_batches}"
        )

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        if self.seed is not None:
            np_rng = np.random.RandomState(self.seed)
        else:
            np_rng = np.random

        for _ in range(self._num_batches):
            batch_indices = []
            used_chains = set()

            # --- 1. Pick anchor subtopic ---
            # Use mean chain weight (not sum) so large subtopics don't
            # dominate purely by having more chains.
            subtopic_weights = np.array([
                np.mean([
                    self.chain_index.static_weight[c]
                    * (self.chain_index.difficulty_ema[c] ** self.chain_index.gamma)
                    for c in chains
                ])
                for chains in self.chain_index.subtopic_to_chains.values()
            ])
            total = subtopic_weights.sum()
            if total > 0:
                subtopic_weights /= total
            else:
                subtopic_weights = np.ones(len(subtopic_weights)) / len(subtopic_weights)

            anchor_idx = np_rng.choice(len(self.chain_index.all_subtopic_keys), p=subtopic_weights)
            anchor_key = self.chain_index.all_subtopic_keys[anchor_idx]
            anchor_ds, anchor_subtopic = anchor_key

            # Find the topic for this subtopic
            anchor_topic = None
            for cid in self.chain_index.subtopic_to_chains[anchor_key]:
                anchor_topic = self.chain_index.chain_to_topic[cid]
                break

            # --- 2. Sample 60% from anchor subtopic ---
            subtopic_chains = self.chain_index.subtopic_to_chains[anchor_key]
            n_sub = min(self.n_subtopic, len(subtopic_chains))
            if n_sub > 0:
                weights = self.chain_index.get_chain_weights(subtopic_chains)
                chosen = np_rng.choice(
                    subtopic_chains, size=n_sub, replace=False, p=weights
                )
                for cid in chosen:
                    used_chains.add(cid)
                    self._sample_k_from_chain(cid, batch_indices, rng)

            # --- 3. Sample 20% from same topic, different subtopic ---
            topic_key = (anchor_ds, anchor_topic)
            if topic_key in self.chain_index.topic_to_chains:
                topic_chains = [
                    c for c in self.chain_index.topic_to_chains[topic_key]
                    if c not in used_chains
                ]
            else:
                topic_chains = []

            n_top = min(self.n_topic, len(topic_chains))
            if n_top > 0:
                weights = self.chain_index.get_chain_weights(topic_chains)
                chosen = np_rng.choice(
                    topic_chains, size=n_top, replace=False, p=weights
                )
                for cid in chosen:
                    used_chains.add(cid)
                    self._sample_k_from_chain(cid, batch_indices, rng)

            # --- 4. Sample 20% random (any dataset/topic) ---
            random_pool = [
                c for c in self.chain_index.all_chains if c not in used_chains
            ]
            n_rand = min(self.n_random, len(random_pool))
            if n_rand > 0:
                weights = self.chain_index.get_chain_weights(random_pool)
                chosen = np_rng.choice(
                    random_pool, size=n_rand, replace=False, p=weights
                )
                for cid in chosen:
                    used_chains.add(cid)
                    self._sample_k_from_chain(cid, batch_indices, rng)

            # --- 5. Backfill PK slots if subtopic/topic didn't have enough ---
            n_pk_filled = len(batch_indices) // self.k
            n_pk_shortfall = self.n_chains - n_pk_filled
            if n_pk_shortfall > 0:
                extra_pool = [
                    c for c in self.chain_index.all_chains if c not in used_chains
                ]
                n_extra = min(n_pk_shortfall, len(extra_pool))
                if n_extra > 0:
                    weights = self.chain_index.get_chain_weights(extra_pool)
                    chosen = np_rng.choice(
                        extra_pool, size=n_extra, replace=False, p=weights
                    )
                    for cid in chosen:
                        used_chains.add(cid)
                        self._sample_k_from_chain(cid, batch_indices, rng)

            # --- 6. Fill remaining with singletons ---
            n_remaining = self.batch_size - len(batch_indices)
            if n_remaining > 0:
                singleton_indices = self.singleton_pool.sample(n_remaining)
                batch_indices.extend(singleton_indices)

            yield batch_indices

    def _sample_k_from_chain(self, chain_id, batch_indices, rng):
        """Add ALL mentions from a chain to batch_indices."""
        batch_indices.extend(self.chain_index.chain_to_indices[chain_id])


# ---------------------------------------------------------------------------
# Custom tokenizer (adapted from cl-coref for standalone use)
# ---------------------------------------------------------------------------

def custom_tokenize_with_special_tokens(
    tokenizer,
    doc_context: str,
    mention_context: str,
    max_length: int = 128,
) -> dict:
    """
    Custom tokenization that preserves [E] and [/E] markers even when truncating.

    Strategy:
    1. Tokenize mention_context separately to find [E] and [/E] positions
    2. If mention_context is too long, truncate it but preserve [E] markers
    3. Allocate remaining space to doc_context
    4. Combine: <s> doc_tokens </s> </s> mention_tokens </s>

    Adapted from cl-coref/src/cl_coref/pipelines/03_training/nodes.py
    """
    special_tokens_count = 4  # <s> ... </s> </s> ... </s>
    max_tokens_for_text = max_length - special_tokens_count

    mention_tokens = tokenizer.encode(mention_context, add_special_tokens=False)

    e_start_id = tokenizer.convert_tokens_to_ids("[E]")
    e_end_id = tokenizer.convert_tokens_to_ids("[/E]")

    try:
        e_start_pos = mention_tokens.index(e_start_id)
        e_end_pos = mention_tokens.index(e_end_id)
    except ValueError:
        return tokenizer(
            text=doc_context,
            text_pair=mention_context,
            padding="max_length",
            max_length=max_length,
            truncation="longest_first",
            return_tensors=None,
        )

    mention_len = len(mention_tokens)

    if mention_len <= max_tokens_for_text:
        doc_max_len = max_tokens_for_text - mention_len
        doc_tokens = tokenizer.encode(doc_context, add_special_tokens=False)
        if len(doc_tokens) > doc_max_len:
            doc_tokens = doc_tokens[:doc_max_len]
    else:
        min_doc_tokens = min(20, max_tokens_for_text // 4)
        mention_max_len = max_tokens_for_text - min_doc_tokens

        tokens_before_e = e_start_pos
        tokens_in_e = e_end_pos - e_start_pos + 1
        tokens_after_e = mention_len - e_end_pos - 1

        if tokens_in_e < mention_max_len:
            remaining_space = mention_max_len - tokens_in_e
            keep_before = min(tokens_before_e, remaining_space // 2)
            keep_after = min(tokens_after_e, remaining_space - keep_before)
            if keep_after < tokens_after_e:
                keep_before = min(tokens_before_e, remaining_space - keep_after)

            start_idx = e_start_pos - keep_before
            end_idx = e_end_pos + 1 + keep_after
            mention_tokens = mention_tokens[start_idx:end_idx]
        else:
            keep_start = mention_max_len // 2
            keep_end = mention_max_len - keep_start
            mention_tokens = (
                mention_tokens[e_start_pos : e_start_pos + keep_start]
                + mention_tokens[e_end_pos + 1 - keep_end : e_end_pos + 1]
            )

        doc_max_len = max_tokens_for_text - len(mention_tokens)
        doc_tokens = tokenizer.encode(doc_context, add_special_tokens=False)
        if len(doc_tokens) > doc_max_len:
            doc_tokens = doc_tokens[:doc_max_len]

    input_ids = (
        [tokenizer.bos_token_id]
        + doc_tokens
        + [tokenizer.eos_token_id]
        + [tokenizer.eos_token_id]
        + mention_tokens
        + [tokenizer.eos_token_id]
    )

    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)

    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    elif padding_length < 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Factory — build everything from the processed parquet
# ---------------------------------------------------------------------------

def build_dataloader(
    mention_df,
    tokenizer,
    batch_size=2048,
    max_length=128,
    tau=1.0,
    gamma=0.5,
    singleton_frac=0.20,
    num_workers=4,
    seed=None,
):
    """
    Build a DataLoader with PK batch sampling from processed mention data.

    All mentions from each sampled chain are included in the batch (no K cap).
    Chains with < 2 mentions are excluded to guarantee positive pairs.

    Args:
        mention_df: Full DataFrame from ucdcr_mentions_processed.parquet.
        tokenizer: HuggingFace tokenizer with [E]/[/E] tokens added.
        batch_size: Target total mentions per batch (actual varies by chain sizes).
        max_length: Tokenizer max length.
        tau: PD exponent for chain weighting.
        gamma: Difficulty EMA exponent.
        singleton_frac: Fraction of batch for singletons.
        num_workers: DataLoader workers.
        seed: Random seed.

    Returns:
        DataLoader, ChainIndex (for updating difficulty during training).
    """
    from torch.utils.data import DataLoader

    # Filter to train split and reset index so dataset, chain index,
    # and singleton pool all share the same 0-based index space.
    train_df = mention_df[mention_df["split"] == "train"].reset_index(drop=True)
    logger.info(f"Training mentions: {len(train_df):,}")

    # Separate singletons from non-singletons (keep reset index)
    singleton_mask = train_df["is_singleton"] == True
    singleton_df = train_df[singleton_mask]
    chain_df = train_df[~singleton_mask]

    # Keep all chains with >= 2 mentions; _sample_k_from_chain caps at chain size
    chain_counts = chain_df.groupby("coref_chain").size()
    valid_chains = chain_counts[chain_counts >= 2].index
    chain_df = chain_df[chain_df["coref_chain"].isin(valid_chains)]

    logger.info(
        f"  Non-singleton: {len(chain_df):,} mentions in "
        f"{chain_df['coref_chain'].nunique():,} chains (>= 2 mentions, sampling up to k={k})"
    )
    logger.info(f"  Singletons: {len(singleton_df):,}")

    # Build index structures (indices match the reset train_df)
    chain_index = ChainIndex(chain_df, tau=tau, gamma=gamma)
    singleton_pool = SingletonPool(singleton_df)

    # Build dataset over ALL training mentions (singletons + chains)
    # so indices from both pools are valid
    dataset = MentionDataset(train_df, tokenizer, max_length=max_length)

    # Build batch sampler
    batch_sampler = PKBatchSampler(
        chain_index=chain_index,
        singleton_pool=singleton_pool,
        batch_size=batch_size,
        singleton_frac=singleton_frac,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, chain_index
