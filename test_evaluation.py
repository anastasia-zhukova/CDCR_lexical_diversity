"""
Test evaluation pipeline on completed inference results.
"""
import sys
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "cl-coref" / "src"))

# Import evaluation function directly to avoid kedro dependency
sys.path.insert(0, str(Path(__file__).parent / "cl-coref" / "src" / "cl_coref" / "pipelines" / "05_evaluation"))
import nodes as eval_nodes
evaluate_all_inference_results = eval_nodes.evaluate_all_inference_results

# Load mentions data
print("Loading mentions data...")
mentions_df = pd.read_csv(
    "cl-coref/data/03_primary/mentions_for_pairs.csv",
    sep=';',
    low_memory=False
)
print(f"Loaded {len(mentions_df):,} mentions")

# Run evaluation
output_base_dir = "cl-coref/data/07_model_output"

print("\n" + "="*80)
print("Running Evaluation Pipeline")
print("="*80)

try:
    results = evaluate_all_inference_results(
        mentions_df=mentions_df,
        output_base_dir=output_base_dir
    )

    metrics_df = results["metrics_summary"]

    if len(metrics_df) > 0:
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)

        # Save results
        output_path = "cl-coref/data/08_reporting/metrics_summary.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print summary
        print(f"\nTotal evaluations: {len(metrics_df)}")

        # Filter to CoNLL metric for summary
        conll_df = metrics_df[metrics_df['metric'] == 'conll']

        if len(conll_df) > 0:
            print(f"\nCoNLL F1 by configuration:")
            summary = conll_df.groupby('training_config').agg({
                'F1': ['mean', 'count']
            }).round(2)
            print(summary)

            # Print top results
            print(f"\nTop 10 Results (by CoNLL F1):")
            top_results = conll_df.nlargest(10, 'F1')[
                ['training_config', 'dataset', 'mode', 'F1', 'Precision', 'Recall']
            ]
            print(top_results.to_string(index=False))

        # Print metrics breakdown
        print(f"\nMetrics available:")
        print(metrics_df['metric'].value_counts())

    else:
        print("\n✗ No evaluation results generated")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
