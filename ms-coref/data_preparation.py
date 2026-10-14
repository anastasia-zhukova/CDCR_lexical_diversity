"""
Data preparation for MS Loss coreference training.

Downloads uCDCR dataset, processes mention contexts into two fields:
  - mention_sentence_context: the mention sentence with [E]/[/E] markers (str)
  - document_context: surrounding sentences without markers (str)

Also computes phrasing diversity (PD) components per mention:
  - mention_wo_stopwords: content words from tokens_text (stopwords + punctuation removed)
  - mention_wordset: frozenset of mention_wo_stopwords (as sorted comma-separated string)
  - mention_head_computed: head word derived from tokens_text (for validation against dataset's mention_head)

These are the validated text representations ready for on-the-fly tokenization.

Output: a single parquet file with all mentions and their representations.
"""

import ast
import logging
import string
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

from core.metrics import phrasing_diversity_calc

# nltk English stopwords (fixed list, avoids nltk dependency)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
    "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn",
    "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't",
}
PUNCTUATION = set(string.punctuation)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "AnZhu/uCDCR"
DATASETS = [
    "CD2CR", "CEREC", "ECBplus", "ECBplusMETAm", "FCC", "GVC",
    "HyperCoref", "MEANTIME", "NIdent", "NP4E", "NewsWCL50", "WECEng",
]

# Columns to keep in the final output (metadata for PK sampling + evaluation)
METADATA_COLS = [
    "mention_id",
    "coref_chain",
    "dataset",
    "split",
    "topic",
    "topic_id",
    "subtopic",
    "subtopic_id",
    "doc_id",
    "conll_doc_key",
    "sent_id",
    "tokens_str",
    "tokens_text",
    "mention_type",
    "mention_full_type",
    "mention_head",
    "mention_head_lemma",
    "mention_ner",
    "is_singleton",
]


def parse_list(val):
    """Parse a string representation of a list, or return as-is if already a list."""
    if isinstance(val, list):
        return val
    return ast.literal_eval(val)


def build_mention_representation(row):
    """
    Build mention_sentence_context (with [E]/[/E] markers) and document_context
    from a uCDCR mention row.

    Handles edge cases where mention_sentence_start_end is too narrow
    (e.g. 29 NewsWCL50 mentions where a trailing quote falls outside sent_end).

    Returns:
        mention_sentence_context: str — mention sentence with [E] mention [/E] markers
        document_context: str — surrounding context (before + after mention sentence)
    """
    ctx = parse_list(row["mention_context"])
    sent_start, sent_end = parse_list(row["mention_sentence_start_end"])
    positions = parse_list(row["tokens_number_context"])

    # Extend sentence window if mention positions fall outside it
    max_mention_pos = max(positions)
    min_mention_pos = min(positions)
    if max_mention_pos >= sent_end:
        sent_end = max_mention_pos + 1
    if min_mention_pos < sent_start:
        sent_start = min_mention_pos

    # Split into mention sentence and document context
    mention_sent_tokens = ctx[sent_start:sent_end]
    doc_ctx_tokens = ctx[:sent_start] + ctx[sent_end:]

    # Compute mention positions relative to the sentence
    positions_in_sent = [p - sent_start for p in positions]

    # Insert [E]/[/E] markers
    marked = mention_sent_tokens.copy()
    marked.insert(positions_in_sent[-1] + 1, "[/E]")
    marked.insert(positions_in_sent[0], "[E]")

    mention_sentence_context = " ".join(marked)
    document_context = " ".join(doc_ctx_tokens)

    return mention_sentence_context, document_context


def compute_pd_components(row):
    """
    Compute phrasing diversity components for a mention, replicating
    the logic from old-approaches/cl-coref-legacy/dataset.py Mention.__init__.

    From tokens_text, removes stopwords and punctuation (keeping the head word
    even if it's a stopword). Produces:
      - mention_wo_stopwords: list of content words (as JSON string for parquet)
      - mention_wordset: sorted unique content words (as comma-separated string)

    Args:
        row: DataFrame row with 'tokens_text' (str repr of list) and 'mention_head' (str)

    Returns:
        mention_wo_stopwords (str), mention_wordset (str)
    """
    tokens_text = parse_list(row["tokens_text"])
    mention_head = row["mention_head"]

    mention_wo_stopwords = [
        token for token in tokens_text
        if token == mention_head
        or (token not in PUNCTUATION and token.lower() not in STOPWORDS)
    ]

    if not mention_wo_stopwords:
        return None, None

    wordset = sorted(set(mention_wo_stopwords))
    return str(mention_wo_stopwords), ",".join(wordset)


def build_chain_table(mention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a chain-level summary table with phrasing diversity and metadata.

    Args:
        mention_df: Mention-level DataFrame (output of download_and_process).

    Returns:
        DataFrame with one row per (dataset, coref_chain).
    """
    logger.info("Building chain-level table...")
    records = []

    for (dataset, chain), grp in mention_df.groupby(["dataset", "coref_chain"]):
        # Size and split distribution
        chain_size = len(grp)
        split_counts = grp["split"].value_counts()
        n_train = int(split_counts.get("train", 0))
        n_val = int(split_counts.get("val", 0) + split_counts.get("validation", 0))
        n_test = int(split_counts.get("test", 0))

        # Cross-document spread
        n_documents = grp["doc_id"].nunique()
        n_sentences = grp.groupby("doc_id")["sent_id"].nunique().sum()

        # Mention types
        mention_types = sorted(grp["mention_type"].dropna().unique().tolist())

        # Unique heads
        heads = grp["mention_head"].dropna().unique().tolist()

        # Phrasing diversity
        pd_mentions = []
        for _, row in grp.iterrows():
            if pd.notna(row["mention_wo_stopwords"]) and pd.notna(row["mention_wordset"]):
                pd_mentions.append((
                    row["mention_head"],
                    row["mention_wordset"],
                    row["mention_wo_stopwords"],
                ))
        pd_score = phrasing_diversity_calc(pd_mentions) if pd_mentions else None

        # Unique wordsets count
        n_unique_wordsets = grp["mention_wordset"].dropna().nunique()

        # Topic / subtopic (take first non-null)
        topic = grp["topic"].iloc[0] if "topic" in grp.columns else None
        subtopic = grp["subtopic"].iloc[0] if "subtopic" in grp.columns else None

        records.append({
            "dataset": dataset,
            "coref_chain": chain,
            "topic": topic,
            "subtopic": subtopic,
            "chain_size": chain_size,
            "is_singleton": chain_size == 1,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_documents": n_documents,
            "n_sentences": int(n_sentences),
            "phrasing_diversity": pd_score,
            "n_unique_heads": len(heads),
            "n_unique_wordsets": n_unique_wordsets,
            "unique_heads": ",".join(sorted(heads)) if heads else None,
            "mention_types": ",".join(mention_types) if mention_types else None,
        })

    chain_df = pd.DataFrame(records)
    logger.info(f"  {len(chain_df):,} chains built")
    return chain_df


def download_and_process(output_dir: str = "data") -> pd.DataFrame:
    """
    Download all uCDCR datasets and create the processed mention representations.

    Args:
        output_dir: Directory to save the output parquet file.

    Returns:
        DataFrame with all mentions, metadata, and text representations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download and combine all datasets
    logger.info("Downloading uCDCR datasets...")
    dfs = []
    for name in DATASETS:
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{name}/all_mentions.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(path)
            df["dataset"] = name
            dfs.append(df)
            logger.info(f"  {name:20s}  {len(df):>8,} mentions")
        except Exception as e:
            logger.error(f"  {name:20s}  FAILED: {e}")

    all_mentions = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total mentions loaded: {len(all_mentions):,}")

    # Build mention representations
    logger.info("Building mention representations...")
    results = all_mentions.apply(build_mention_representation, axis=1, result_type="expand")
    results.columns = ["mention_sentence_context", "document_context"]
    all_mentions = pd.concat([all_mentions, results], axis=1)

    # Validate: check [E] and [/E] are present in all mention_sentence_context
    has_e_start = all_mentions["mention_sentence_context"].str.contains(r"\[E\]", regex=True)
    has_e_end = all_mentions["mention_sentence_context"].str.contains(r"\[/E\]", regex=True)
    n_missing = (~has_e_start | ~has_e_end).sum()
    if n_missing > 0:
        logger.warning(f"  {n_missing} mentions missing [E]/[/E] markers!")
    else:
        logger.info(f"  All {len(all_mentions):,} mentions have [E]/[/E] markers")

    # Compute phrasing diversity components
    logger.info("Computing phrasing diversity components...")
    pd_results = all_mentions.apply(compute_pd_components, axis=1, result_type="expand")
    pd_results.columns = ["mention_wo_stopwords", "mention_wordset"]
    all_mentions = pd.concat([all_mentions, pd_results], axis=1)

    n_null_pd = all_mentions["mention_wo_stopwords"].isna().sum()
    logger.info(f"  PD components: {len(all_mentions) - n_null_pd:,} computed, "
                f"{n_null_pd:,} null (head word not in tokens after filtering)")
    if n_null_pd > 0:
        null_examples = all_mentions[all_mentions["mention_wo_stopwords"].isna()][
            ["dataset", "mention_id", "tokens_str", "mention_head"]
        ].head(5)
        for _, ex in null_examples.iterrows():
            logger.info(f"    null PD: [{ex['dataset']}] \"{ex['tokens_str']}\" "
                        f"(head=\"{ex['mention_head']}\")")

    # Validate mention_head: check that dataset's mention_head is in tokens_text
    logger.info("Validating mention_head presence in tokens_text...")
    head_in_tokens = all_mentions.apply(
        lambda r: r["mention_head"] in parse_list(r["tokens_text"]), axis=1
    )
    n_head_missing = (~head_in_tokens).sum()
    if n_head_missing > 0:
        logger.warning(f"  mention_head NOT in tokens_text: {n_head_missing:,} mentions")
        # Breakdown by dataset
        all_mentions["_head_in_tokens"] = head_in_tokens
        head_issues = all_mentions[~all_mentions["_head_in_tokens"]].groupby("dataset").size()
        for ds, count in head_issues.items():
            logger.warning(f"    {ds}: {count}")
        # Show examples
        examples = all_mentions[~all_mentions["_head_in_tokens"]][
            ["dataset", "mention_id", "tokens_str", "mention_head", "tokens_text"]
        ].head(10)
        for _, ex in examples.iterrows():
            tokens = parse_list(ex["tokens_text"])
            logger.warning(f"    [{ex['dataset']}] \"{ex['tokens_str']}\" "
                           f"head=\"{ex['mention_head']}\" tokens={tokens}")
        all_mentions.drop(columns=["_head_in_tokens"], inplace=True)
    else:
        logger.info(f"  All {len(all_mentions):,} mentions have mention_head in tokens_text")

    # Select output columns
    output_cols = METADATA_COLS + [
        "mention_sentence_context", "document_context",
        "mention_wo_stopwords", "mention_wordset",
    ]
    # Only keep columns that exist (FCC may be missing some)
    available_cols = [c for c in output_cols if c in all_mentions.columns]
    missing_cols = set(output_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"  Missing columns (will be null): {missing_cols}")
        for col in missing_cols:
            all_mentions[col] = None
        available_cols = output_cols

    result_df = all_mentions[available_cols].copy()

    # Normalize mixed-type columns for parquet compatibility
    for col in ["topic_id", "topic", "subtopic", "subtopic_id", "coref_chain"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype(str)

    # Save
    out_file = output_path / "ucdcr_mentions_processed.parquet"
    result_df.to_parquet(out_file, index=False)
    logger.info(f"Saved to {out_file} ({len(result_df):,} rows)")

    # Print summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Datasets:   {result_df['dataset'].nunique()}")
    logger.info(f"  Mentions:   {len(result_df):,}")
    logger.info(f"  Singletons: {result_df['is_singleton'].sum():,} "
                f"({result_df['is_singleton'].mean()*100:.1f}%)")

    for split in ["train", "val", "validation", "test"]:
        n = (result_df["split"] == split).sum()
        if n > 0:
            logger.info(f"  Split '{split}': {n:,}")

    logger.info("\n  Per dataset:")
    for ds in sorted(result_df["dataset"].unique()):
        ds_df = result_df[result_df["dataset"] == ds]
        chains = ds_df["coref_chain"].nunique()
        non_single = ds_df[~ds_df["is_singleton"]]
        chain_sizes = non_single.groupby("coref_chain").size()
        avg_size = chain_sizes.mean() if len(chain_sizes) > 0 else 0
        logger.info(f"    {ds:20s}  {len(ds_df):>8,} mentions  "
                    f"{chains:>6,} chains  avg_size={avg_size:.1f}")

    # Build and save chain-level table
    chain_df = build_chain_table(result_df)
    chain_file = output_path / "ucdcr_chains.parquet"
    chain_df.to_parquet(chain_file, index=False)
    logger.info(f"Saved chain table to {chain_file} ({len(chain_df):,} rows)")

    # Chain-level summary
    logger.info("\nChain-level summary:")
    logger.info(f"  Total chains:     {len(chain_df):,}")
    logger.info(f"  Singletons:       {chain_df['is_singleton'].sum():,} "
                f"({chain_df['is_singleton'].mean()*100:.1f}%)")
    non_single_chains = chain_df[~chain_df["is_singleton"]]
    logger.info(f"  Non-singleton:    {len(non_single_chains):,}")
    logger.info(f"  Avg chain size:   {chain_df['chain_size'].mean():.1f}")
    logger.info(f"  Avg non-single:   {non_single_chains['chain_size'].mean():.1f}")
    logger.info(f"  Avg PD (non-singleton): {non_single_chains['phrasing_diversity'].dropna().mean():.3f}")
    logger.info(f"  Cross-doc chains: {(chain_df['n_documents'] > 1).sum():,} "
                f"({(chain_df['n_documents'] > 1).mean()*100:.1f}%)")

    return result_df


if __name__ == "__main__":
    download_and_process()
