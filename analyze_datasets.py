#!/usr/bin/env python3
"""
Simple script to analyze CDCR datasets without loading them into memory.
Reads parquet files directly to get statistics.
"""

import os
import sys
import pandas as pd
import json
from collections import defaultdict
import time

BASE_DATA_PATH = "CL_COREF/data"

DATASETS = [
    "CD2CR",
    "CEREC",
    "ECBplus",
    "ECBplusMETAm",
    "FCC",
    "GVC",
    "HyperCoref",
    "NP4E",
    "NewsWCL50",
    "NiDENT",
    "WECEng"
]

def estimate_pairs_from_clusters(cluster_sizes):
    """
    Estimate number of positive and negative pairs based on cluster sizes.

    Positive pairs: n*(n-1)/2 for each cluster of size n
    Negative pairs: estimated as 2x positive (typical ratio)
    """
    # Positive pairs: sum of n*(n-1)/2 for each cluster
    positive_pairs = sum(n * (n - 1) // 2 for n in cluster_sizes if n > 1)

    # Negative pairs: estimate based on typical sampling ratios
    negative_pairs = positive_pairs * 2

    total_pairs = positive_pairs + negative_pairs

    return total_pairs, positive_pairs, negative_pairs

def analyze_dataset(dataset_name, split="train"):
    """Analyze a single dataset by reading files directly."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name} ({split})")
    print(f"{'='*70}")

    results = {
        'dataset_name': dataset_name,
        'split': split,
    }

    try:
        split_path = os.path.join(BASE_DATA_PATH, dataset_name, split)
        parent_path = os.path.join(BASE_DATA_PATH, dataset_name)

        if not os.path.exists(split_path):
            print(f"⚠ Warning: Path does not exist: {split_path}")
            return None

        # Read parquet file for documents
        parquet_path = os.path.join(parent_path, "all_documents.parquet")
        if not os.path.exists(parquet_path):
            print(f"⚠ Warning: Parquet file not found: {parquet_path}")
            return None

        print("Loading documents parquet...")
        start_time = time.time()
        df_docs = pd.read_parquet(parquet_path)
        load_time = time.time() - start_time
        print(f"  ✓ Loaded {len(df_docs)} document rows in {load_time:.2f}s")

        # Count documents and topics
        num_docs = df_docs['topic/subtopic_name/doc'].nunique()
        topics = df_docs['topic/subtopic_name/doc'].str.split('/').str[0].unique()
        num_topics = len(topics)

        results['num_documents'] = num_docs
        results['num_topics'] = num_topics
        results['topics'] = sorted(topics.tolist())

        # Load mentions
        total_mentions = 0
        entity_mentions = 0
        event_mentions = 0
        mentions_by_cluster = defaultdict(int)

        for mention_file, mention_type in [('entity_mentions.json', 'entity'),
                                           ('event_mentions.json', 'event')]:
            mention_path = os.path.join(split_path, mention_file)
            if os.path.exists(mention_path):
                print(f"Loading {mention_file}...")
                with open(mention_path, 'r') as f:
                    mentions = json.load(f)

                print(f"  ✓ Found {len(mentions)} {mention_type} mentions")

                if mention_type == 'entity':
                    entity_mentions = len(mentions)
                else:
                    event_mentions = len(mentions)

                total_mentions += len(mentions)

                # Count cluster sizes
                for m in mentions:
                    if 'coref_chain' in m:
                        mentions_by_cluster[m['coref_chain']] += 1

        results['total_mentions'] = total_mentions
        results['entity_mentions'] = entity_mentions
        results['event_mentions'] = event_mentions

        if total_mentions == 0:
            print(f"⚠ No mentions found")
            return None

        # Cluster statistics
        cluster_sizes = list(mentions_by_cluster.values())
        num_clusters = len(cluster_sizes)
        singletons = sum(1 for size in cluster_sizes if size == 1)
        non_singleton_clusters = sum(1 for size in cluster_sizes if size > 1)
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0

        results['num_clusters'] = num_clusters
        results['singletons'] = singletons
        results['non_singleton_clusters'] = non_singleton_clusters
        results['avg_cluster_size'] = avg_cluster_size
        results['max_cluster_size'] = max_cluster_size

        # Estimate pairs
        total_pairs, positive_pairs, negative_pairs = estimate_pairs_from_clusters(cluster_sizes)
        results['estimated_total_pairs'] = total_pairs
        results['estimated_positive_pairs'] = positive_pairs
        results['estimated_negative_pairs'] = negative_pairs
        results['pairs_per_mention'] = total_pairs / total_mentions if total_mentions > 0 else 0

        # Print summary
        print(f"\n{'─'*70}")
        print(f"STATISTICS:")
        print(f"  Mentions: {total_mentions:,} (entity: {entity_mentions:,}, event: {event_mentions:,})")
        print(f"  Documents: {num_docs:,}")
        print(f"  Topics: {num_topics}")
        print(f"  Clusters: {num_clusters:,} ({non_singleton_clusters:,} non-singleton, {singletons:,} singletons)")
        print(f"  Avg cluster size: {avg_cluster_size:.2f}")
        print(f"  Max cluster size: {max_cluster_size}")
        print(f"\nESTIMATED PAIRS:")
        print(f"  Total: ~{total_pairs:,}")
        print(f"  Positive: {positive_pairs:,}")
        print(f"  Negative: ~{negative_pairs:,} (estimate)")
        print(f"  Pairs per mention: {results['pairs_per_mention']:.2f}")
        print(f"{'─'*70}")

        return results

    except Exception as e:
        print(f"✗ Error analyzing {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("CDCR Dataset Analysis")
    print("="*70)
    print(f"Analyzing {len(DATASETS)} datasets...")

    all_results = []

    for dataset_name in DATASETS:
        result = analyze_dataset(dataset_name, split="train")
        if result:
            all_results.append(result)

    print(f"\n\n{'='*70}")
    print(f"SUMMARY: Processed {len(all_results)}/{len(DATASETS)} datasets")
    print(f"{'='*70}")

    if all_results:
        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Print summary table
        print("\nRESULTS TABLE:")
        print("="*120)
        summary_cols = ['dataset_name', 'total_mentions', 'num_topics', 'num_clusters',
                       'estimated_total_pairs', 'estimated_positive_pairs', 'pairs_per_mention']
        print(df[summary_cols].to_string(index=False))

        # Save to CSV
        output_file = "dataset_analysis_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Full results saved to: {output_file}")

        # Print key findings
        print(f"\n{'='*70}")
        print("KEY FINDINGS:")
        print(f"{'='*70}")

        total_mentions = df['total_mentions'].sum()
        total_pairs = df['estimated_total_pairs'].sum()
        avg_pairs_per_mention = df['pairs_per_mention'].mean()

        print(f"\n1. TOTALS:")
        print(f"   - Datasets: {len(df)}")
        print(f"   - Mentions: {total_mentions:,}")
        print(f"   - Estimated pairs: {total_pairs:,}")
        print(f"   - Avg pairs/mention: {avg_pairs_per_mention:.2f}")

        print(f"\n2. LARGEST DATASETS:")
        for _, row in df.nlargest(3, 'total_mentions').iterrows():
            print(f"   - {row['dataset_name']}: {row['total_mentions']:,} mentions, ~{row['estimated_total_pairs']:,} pairs")

        print(f"\n3. MOST PAIRS:")
        for _, row in df.nlargest(3, 'estimated_total_pairs').iterrows():
            print(f"   - {row['dataset_name']}: ~{row['estimated_total_pairs']:,} pairs ({row['pairs_per_mention']:.2f}/mention)")

        # Suggest balanced configs
        median_mentions = df['total_mentions'].median()
        print(f"\n4. SUGGESTED BALANCED CONFIGS (median: {int(median_mentions)} mentions):")

        setup_id = 10
        for _, row in df.iterrows():
            if row['total_mentions'] > median_mentions * 1.5:
                suggested_max = int(median_mentions)
                print(f"\n{setup_id}:")
                print(f"  name: \"{row['dataset_name'].lower()}_balanced\"")
                print(f"  datasets:")
                print(f"    - name: \"{row['dataset_name']}\"")
                print(f"      splits: [\"train\", \"val\"]")
                print(f"      topic_filter: null")
                print(f"      max_mentions: {suggested_max}  # Original: {row['total_mentions']:,}")
                setup_id += 1

        print(f"\n{'='*70}")
        print("✓ Analysis complete!")
        print(f"{'='*70}")
    else:
        print("\n⚠ No datasets were successfully analyzed")

if __name__ == "__main__":
    main()
