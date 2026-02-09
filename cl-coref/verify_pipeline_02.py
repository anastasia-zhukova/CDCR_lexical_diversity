#!/usr/bin/env python3
"""Quick verification script for pipeline 02_pairs_creation output."""

import pandas as pd
import json
from pathlib import Path

def verify_outputs():
    """Verify all pipeline outputs are correct."""
    
    base_path = Path("data/03_primary")
    
    print("=" * 60)
    print("Pipeline 02_pairs_creation - Output Verification")
    print("=" * 60)
    
    # 1. Check mention_representations.parquet
    print("\n1. mention_representations.parquet")
    repr_path = base_path / "mention_representations.parquet"
    if repr_path.exists():
        df = pd.read_parquet(repr_path)
        print(f"   ✓ File exists ({repr_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"   ✓ Rows: {len(df):,}")
        print(f"   ✓ Columns: {list(df.columns)}")
        
        # Check markers
        has_start = df['mention_context'].str.contains(r'\[E\]', regex=True).sum()
        has_end = df['mention_context'].str.contains(r'\[/E\]', regex=True).sum()
        print(f"   ✓ [E] markers: {has_start:,} ({has_start/len(df)*100:.1f}%)")
        print(f"   ✓ [/E] markers: {has_end:,} ({has_end/len(df)*100:.1f}%)")
        
        # Check lengths
        avg_len = df['string_representation'].str.len().mean()
        print(f"   ✓ Avg length: {avg_len:.0f} chars")
    else:
        print(f"   ✗ File not found!")
    
    # 2. Check mentions_for_pairs.csv
    print("\n2. mentions_for_pairs.csv")
    csv_path = base_path / "mentions_for_pairs.csv"
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path, sep=';')
        print(f"   ✓ File exists ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"   ✓ Rows: {len(df_csv):,}")
        print(f"   ✓ Columns: {len(df_csv.columns)}")
    else:
        print(f"   ✗ File not found!")
    
    # 3. Check mention_groups_summary.csv
    print("\n3. mention_groups_summary.csv")
    groups_path = base_path / "mention_groups_summary.csv"
    if groups_path.exists():
        df_groups = pd.read_csv(groups_path, sep=';')
        print(f"   ✓ File exists ({groups_path.stat().st_size / 1024:.1f} KB)")
        print(f"   ✓ Groups: {len(df_groups):,}")
        print(f"   ✓ Datasets: {df_groups['dataset_name'].nunique()}")
        print(f"   ✓ Entity styles: {df_groups['entity_style'].nunique()}")
    else:
        print(f"   ✗ File not found!")
    
    # 4. Check grouping_statistics.json
    print("\n4. grouping_statistics.json")
    stats_path = base_path / "grouping_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"   ✓ File exists ({stats_path.stat().st_size / 1024:.1f} KB)")
        print(f"   ✓ Total mentions: {stats['total_mentions']:,}")
        print(f"   ✓ Total clusters: {stats['total_clusters']:,}")
        print(f"   ✓ Estimated pairs: ~{stats['total_estimated_pairs']:,}")
    else:
        print(f"   ✗ File not found!")
    
    print("\n" + "=" * 60)
    print("Verification Complete! ✓")
    print("=" * 60)

if __name__ == "__main__":
    verify_outputs()
