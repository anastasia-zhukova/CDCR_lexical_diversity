#!/usr/bin/env python3
"""
Quick test script for the 00_data_exploration pipeline.

This script tests the pipeline nodes independently without running the full Kedro pipeline.
Useful for debugging and verification.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cl_coref.pipelines.00_data_exploration.nodes import (
    load_all_mentions_from_datasets,
    compute_dataset_statistics,
    create_summary_table,
)


def test_load_mentions():
    """Test loading all mentions from datasets."""
    print("=" * 70)
    print("TEST: Loading all mentions")
    print("=" * 70)

    data_path = "data/01_raw/cdcr_datasets"

    if not Path(data_path).exists():
        print(f"⚠️  Data path not found: {data_path}")
        print("Please ensure the data exists at this location")
        return None

    try:
        df = load_all_mentions_from_datasets(data_path)

        print(f"\n✓ Successfully loaded DataFrame")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\n  Datasets: {df['dataset_name'].nunique()}")
        print(f"  Splits: {df['split'].unique().tolist()}")
        print(f"  Entity styles: {df['entity_style'].value_counts().to_dict()}")

        print(f"\n  Sample metadata:")
        print(df[["dataset_name", "split", "entity_style", "mention_id", "tokens_str"]].head(10))

        return df

    except Exception as e:
        print(f"❌ Error loading mentions: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_compute_statistics(df):
    """Test computing statistics."""
    print("\n" + "=" * 70)
    print("TEST: Computing statistics")
    print("=" * 70)

    if df is None or df.empty:
        print("⚠️  No data to compute statistics")
        return None

    try:
        stats = compute_dataset_statistics(df)

        print(f"\n✓ Successfully computed statistics")
        print(f"\n  Total mentions: {stats['total_mentions']:,}")
        print(f"  Datasets: {len(stats['by_dataset'])}")

        print(f"\n  By dataset:")
        for dataset, dataset_stats in list(stats["by_dataset"].items())[:5]:
            print(f"    {dataset}: {dataset_stats['total_mentions']:,} mentions")

        print(f"\n  By split:")
        for split, split_stats in stats["by_split"].items():
            print(f"    {split}: {split_stats['total_mentions']:,} mentions")

        if "cluster_stats" in stats and stats["cluster_stats"]:
            print(f"\n  Cluster stats:")
            print(f"    Total clusters: {stats['cluster_stats']['total_clusters']:,}")
            print(f"    Singletons: {stats['cluster_stats']['singletons']:,}")
            print(f"    Non-singletons: {stats['cluster_stats']['non_singletons']:,}")
            print(f"    Estimated pairs: ~{stats['cluster_stats']['estimated_total_pairs']:,}")

        return stats

    except Exception as e:
        print(f"❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_summary_table(df):
    """Test creating summary table."""
    print("\n" + "=" * 70)
    print("TEST: Creating summary table")
    print("=" * 70)

    if df is None or df.empty:
        print("⚠️  No data to create summary")
        return None

    try:
        summary = create_summary_table(df)

        print(f"\n✓ Successfully created summary table")
        print(f"  Shape: {summary.shape}")
        print(f"\n{summary.to_string(index=False)}")

        return summary

    except Exception as e:
        print(f"❌ Error creating summary: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("=" * 70)
    print("DATA EXPLORATION PIPELINE - TEST")
    print("=" * 70)

    # Test 1: Load mentions
    df = test_load_mentions()

    if df is None or df.empty:
        print("\n❌ Tests failed - could not load data")
        return 1

    # Test 2: Compute statistics
    stats = test_compute_statistics(df)

    # Test 3: Create summary table
    summary = test_summary_table(df)

    # Final summary
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)

    if stats and summary is not None:
        print("✓ All tests passed!")
        print("\nTo run the full pipeline:")
        print("  kedro run --pipeline 00_data_exploration")
        return 0
    else:
        print("⚠️  Some tests had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
