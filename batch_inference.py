"""
Batch inference runner for all trained models.
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Set up simple logging (not Rich)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/batch_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add CL_COREF to path
sys.path.insert(0, str(Path(__file__).parent / "cl-coref" / "src"))

# Import the inference function (module name starts with digit, need importlib)
import importlib
inference_nodes = importlib.import_module('cl_coref.pipelines.04_inference.nodes')
run_full_inference_pipeline = inference_nodes.run_full_inference_pipeline

# Load data once
print("="*80)
print("BATCH INFERENCE - Loading data...")
print("="*80)
mentions_df = pd.read_csv(
    "cl-coref/data/03_primary/mentions_for_pairs.csv",
    sep=';',
    low_memory=False
)
print(f"Loaded {len(mentions_df):,} mentions")

mention_representations = pd.read_parquet(
    "cl-coref/data/03_primary/mention_representations.parquet"
)
print(f"Loaded {len(mention_representations):,} representations\n")

# Define all model configurations
models_base = Path("cl-coref/data/06_models")
output_base_dir = "cl-coref/data/07_model_output"
similarity_threshold = 0.5

# Find all best checkpoints (exclude "last.ckpt" files)
checkpoints = []
for ckpt_path in models_base.glob("*/*.ckpt"):
    if ckpt_path.name != "last.ckpt" and "last-v" not in ckpt_path.name:
        training_config = ckpt_path.parent.name
        checkpoints.append({
            'config': training_config,
            'path': str(ckpt_path)  # Keep as-is, already has correct path
        })

# Sort by config name for consistent ordering
checkpoints.sort(key=lambda x: x['config'])

print(f"Found {len(checkpoints)} trained models:")
for ckpt in checkpoints:
    print(f"  - {ckpt['config']}")
print()

# Run inference for each model
results_summary = []
start_time = datetime.now()

for i, ckpt in enumerate(checkpoints, 1):
    config = ckpt['config']
    model_path = ckpt['path']

    print("\n" + "="*80)
    print(f"[{i}/{len(checkpoints)}] Processing: {config}")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        ckpt_start = datetime.now()

        result = run_full_inference_pipeline(
            mentions_df=mentions_df,
            mention_representations=mention_representations,
            training_config=config,
            model_checkpoint_path=model_path,
            similarity_threshold=similarity_threshold,
            output_base_dir=output_base_dir
        )

        ckpt_duration = (datetime.now() - ckpt_start).total_seconds()

        results_summary.append({
            'config': config,
            'status': 'SUCCESS',
            'duration_seconds': ckpt_duration,
            'datasets_evaluated': result['total_datasets']
        })

        print(f"\n✓ SUCCESS - Duration: {ckpt_duration:.1f}s")

    except Exception as e:
        ckpt_duration = (datetime.now() - ckpt_start).total_seconds()

        results_summary.append({
            'config': config,
            'status': 'FAILED',
            'duration_seconds': ckpt_duration,
            'error': str(e)
        })

        print(f"\n✗ FAILED - Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

# Print final summary
total_duration = (datetime.now() - start_time).total_seconds()

print("\n\n" + "="*80)
print("BATCH INFERENCE COMPLETE")
print("="*80)
print(f"Total duration: {total_duration/60:.1f} minutes")
print(f"Total models: {len(checkpoints)}")
print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'SUCCESS')}")
print(f"Failed: {sum(1 for r in results_summary if r['status'] == 'FAILED')}")

print("\nResults by model:")
for result in results_summary:
    status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
    duration = result['duration_seconds']
    print(f"  {status_icon} {result['config']:<25} - {duration:>6.1f}s - {result['status']}")
    if result['status'] == 'FAILED':
        print(f"      Error: {result.get('error', 'Unknown error')}")

print("\nOutput directory: cl-coref/data/07_model_output/")
print(f"Log file: /tmp/batch_inference.log")
