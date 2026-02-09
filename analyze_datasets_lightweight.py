#!/usr/bin/env python3
"""
Ultra-lightweight script to analyze CDCR datasets.
Only reads JSON mention files, doesn't load documents or use pandas for large files.
"""

import os
import json
from collections import defaultdict

BASE_DATA_PATH = "CL_COREF/data"

DATASETS = [
    "CD2CR", "CEREC", "ECBplus", "ECBplusMETAm", "FCC",
    "GVC", "HyperCoref", "NP4E", "NewsWCL50", "NiDENT", "WECEng"
]

# Topic filters per dataset (None = no filter, load all topics)
TOPIC_FILTERS = {
    "HyperCoref": ["us", "politics", "international"],
    "WECEng": [
        "Meetings",
        "Civilian Attack",
        "Airliner Accident",
        "Earthquake",
        "News Event",
        "Terrorist Attack",
        "Wildfire",
        "Flood",
        "Weapons Test",
        "Eruption",
        "Oilspill",
        "Rail Accident"
    ]
}

def estimate_pairs(cluster_sizes):
    """Estimate pairs from cluster sizes."""
    positive = sum(n * (n - 1) // 2 for n in cluster_sizes if n > 1)
    negative = positive * 2  # Estimate
    return positive + negative, positive, negative

def analyze_dataset_lightweight(dataset_name, split="train", topic_filter=None):
    """Analyze dataset by only reading mention JSON files."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} ({split})")
    if topic_filter:
        print(f"  Topic filter: {', '.join(topic_filter)}")
    print(f"{'='*60}")

    split_path = os.path.join(BASE_DATA_PATH, dataset_name, split)

    if not os.path.exists(split_path):
        print(f"⚠ Path not found")
        return None

    total_mentions = 0
    total_mentions_raw = 0
    entity_count = 0
    entity_count_raw = 0
    event_count = 0
    event_count_raw = 0
    clusters = defaultdict(int)
    topics = set()

    # Read mention files
    for filename, mtype in [('entity_mentions.json', 'entity'),
                            ('event_mentions.json', 'event')]:
        fpath = os.path.join(split_path, filename)
        if not os.path.exists(fpath):
            continue

        try:
            with open(fpath, 'r') as f:
                mentions = json.load(f)

            count_raw = len(mentions)
            count_filtered = 0

            # Collect cluster and topic info (with optional filtering)
            for m in mentions:
                # Get topic ID - try both 'topic' and 'topic_id' keys
                topic_id = str(m.get('topic', m.get('topic_id', '')))
                topics.add(topic_id)

                # Check if this mention passes the filter
                if topic_filter and topic_id not in topic_filter:
                    continue  # Skip this mention

                # This mention passes the filter
                count_filtered += 1

                # Count clusters
                if 'coref_chain' in m:
                    clusters[m['coref_chain']] += 1

            # Update counts
            total_mentions_raw += count_raw
            total_mentions += count_filtered

            if mtype == 'entity':
                entity_count_raw = count_raw
                entity_count = count_filtered
            else:
                event_count_raw = count_raw
                event_count = count_filtered

            # Print with filtering info
            if topic_filter:
                print(f"  {filename}: {count_raw:,} mentions → {count_filtered:,} after filtering")
            else:
                print(f"  {filename}: {count_filtered:,} mentions")

        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue

    if total_mentions == 0:
        print("  ⚠ No mentions")
        return None

    # Compute statistics
    cluster_sizes = list(clusters.values())
    num_clusters = len(cluster_sizes)
    singletons = sum(1 for s in cluster_sizes if s == 1)
    non_singletons = num_clusters - singletons
    avg_size = sum(cluster_sizes) / num_clusters if num_clusters else 0
    max_size = max(cluster_sizes) if cluster_sizes else 0

    total_pairs, pos_pairs, neg_pairs = estimate_pairs(cluster_sizes)
    pairs_per_mention = total_pairs / total_mentions if total_mentions > 0 else 0

    # Print results
    print(f"\nMentions: {total_mentions:,} (entity: {entity_count:,}, event: {event_count:,})")
    print(f"Topics: {len(topics)}")
    if topics:
        sorted_topics = sorted(topics, key=lambda x: (len(x), x))  # Sort by length then value
        topics_str = ', '.join(sorted_topics[:20])  # Show first 20
        if len(sorted_topics) > 20:
            topics_str += f", ... ({len(sorted_topics)-20} more)"
        print(f"  Topic IDs: {topics_str}")
    print(f"Clusters: {num_clusters:,} ({non_singletons:,} non-singleton)")
    print(f"Avg cluster size: {avg_size:.1f}, Max: {max_size}")
    print(f"Estimated pairs: ~{total_pairs:,} ({pos_pairs:,} pos, ~{neg_pairs:,} neg)")
    print(f"Pairs/mention: {pairs_per_mention:.2f}")

    # Sort topics for consistent output
    sorted_topics = sorted(topics, key=lambda x: (len(x), x))

    return {
        'dataset': dataset_name,
        'mentions': total_mentions,
        'entity': entity_count,
        'event': event_count,
        'topics': len(topics),
        'topic_list': ';'.join(sorted_topics),  # Join with semicolon for CSV
        'clusters': num_clusters,
        'non_singleton_clusters': non_singletons,
        'avg_cluster_size': avg_size,
        'max_cluster_size': max_size,
        'est_pairs': total_pairs,
        'est_pos_pairs': pos_pairs,
        'est_neg_pairs': neg_pairs,
        'pairs_per_mention': pairs_per_mention
    }

def main():
    print("="*60)
    print("CDCR Dataset Analysis (Lightweight)")
    print("="*60)

    results = []
    for dataset in DATASETS:
        # Get topic filter for this dataset (if any)
        topic_filter = TOPIC_FILTERS.get(dataset)
        result = analyze_dataset_lightweight(dataset, topic_filter=topic_filter)
        if result:
            results.append(result)

    if not results:
        print("\n⚠ No datasets analyzed")
        return

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY ({len(results)}/{len(DATASETS)} datasets)")
    print(f"{'='*60}\n")

    # Print table
    print(f"{'Dataset':<15} {'Mentions':>10} {'Topics':>7} {'Clusters':>9} {'Est.Pairs':>11} {'Pairs/M':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['dataset']:<15} {r['mentions']:>10,} {r['topics']:>7} "
              f"{r['clusters']:>9,} {r['est_pairs']:>11,} {r['pairs_per_mention']:>8.2f}")

    # Totals
    total_mentions = sum(r['mentions'] for r in results)
    total_pairs = sum(r['est_pairs'] for r in results)
    avg_pairs_per_m = sum(r['pairs_per_mention'] for r in results) / len(results)

    print("-" * 70)
    print(f"{'TOTAL':<15} {total_mentions:>10,} {'':>7} {'':>9} {total_pairs:>11,} {avg_pairs_per_m:>8.2f}")

    # Top 3
    print(f"\nLARGEST (by mentions):")
    for r in sorted(results, key=lambda x: x['mentions'], reverse=True)[:3]:
        print(f"  {r['dataset']}: {r['mentions']:,} mentions, ~{r['est_pairs']:,} pairs")

    print(f"\nMOST PAIRS:")
    for r in sorted(results, key=lambda x: x['est_pairs'], reverse=True)[:3]:
        print(f"  {r['dataset']}: ~{r['est_pairs']:,} pairs ({r['pairs_per_mention']:.2f}/mention)")

    # Save to simple CSV
    try:
        with open('dataset_analysis_simple.csv', 'w') as f:
            # Header
            f.write(','.join(results[0].keys()) + '\n')
            # Data
            for r in results:
                f.write(','.join(str(v) for v in r.values()) + '\n')
        print(f"\n✓ Saved to: dataset_analysis_simple.csv")
    except Exception as e:
        print(f"\n⚠ Could not save CSV: {e}")

    print(f"\n{'='*60}")
    print("✓ Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
