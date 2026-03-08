"""
Direct inference runner that bypasses Kedro logging.
"""
import sys
import pandas as pd
import logging
from pathlib import Path

# Set up simple logging (not Rich)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/inference_direct.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add CL_COREF to path
sys.path.insert(0, str(Path(__file__).parent / "cl-coref" / "src"))

# Import the inference function (module name starts with digit, need importlib)
import importlib
inference_nodes = importlib.import_module('cl_coref.pipelines.04_inference.nodes')
run_full_inference_pipeline = inference_nodes.run_full_inference_pipeline

# Load data
print("Loading data...")
mentions_df = pd.read_csv(
    "cl-coref/data/03_primary/mentions_for_pairs.csv",
    sep=';',
    low_memory=False
)
print(f"Loaded {len(mentions_df):,} mentions")

mention_representations = pd.read_parquet(
    "cl-coref/data/03_primary/mention_representations.parquet"
)
print(f"Loaded {len(mention_representations):,} representations")

# Set parameters for single-ECBplus
training_config = "single-ECBplus"
model_checkpoint_path = "cl-coref/data/06_models/ECBplus/siamese_model_ECBplus_epoch=06_val_f1_conll=44.23.ckpt"
similarity_threshold = 0.5
output_base_dir = "cl-coref/data/07_model_output"

print(f"\nRunning inference for: {training_config}")
print(f"Model: {model_checkpoint_path}")
print(f"Similarity threshold: {similarity_threshold}")
print(f"Output directory: {output_base_dir}")

# Run inference
try:
    results = run_full_inference_pipeline(
        mentions_df=mentions_df,
        mention_representations=mention_representations,
        training_config=training_config,
        model_checkpoint_path=model_checkpoint_path,
        similarity_threshold=similarity_threshold,
        output_base_dir=output_base_dir
    )
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"Results: {results}")
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
