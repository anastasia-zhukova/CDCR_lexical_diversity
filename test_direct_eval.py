"""
Direct test of CoNLL evaluation using the official scorer.
"""
import subprocess
import sys
from pathlib import Path

# Use the official CoNLL scorer directly on our files
key_path = "cl-coref/data/07_model_output/ECBplus/ECBplus/Events/key.conll"
response_path = "cl-coref/data/07_model_output/ECBplus/ECBplus/Events/response.conll"

# Add CL_COREF to path to use coval
sys.path.insert(0, str(Path("CL_COREF")))

from coval.conll.reader import get_coref_infos
from coval.eval import evaluator

print("="*80)
print("Direct CoNLL Evaluation Test")
print("="*80)

# Read the CoNLL files using coval's reader
doc_coref_infos = get_coref_infos(
    key_file=key_path,
    sys_file=response_path,
    NP_only=False,
    remove_nested=False,
    keep_singletons=True,
    min_span=False
)

print(f"\nDocuments found: {len(doc_coref_infos)}")

# Define metrics to compute
metrics = [
    ('muc', evaluator.muc),
    ('bcub', evaluator.b_cubed),
    ('ceafe', evaluator.ceafe),
]

print("\nEvaluating...")
# Custom coval requires dict_phrasing_diversity, dataset_name, mention_type
dict_phrasing_diversity = {}
results = {}
for name, metric in metrics:
    recall, precision, f1 = evaluator.evaluate_documents(
        doc_coref_infos,
        dict_phrasing_diversity,
        metric,
        dataset_name='ECBplus',
        mention_type='all',
        beta=1
    )
    results[name] = {
        'recall': recall * 100,
        'precision': precision * 100,
        'f1': f1 * 100
    }
    print(f"{name.upper():10s}: R={recall*100:.2f}  P={precision*100:.2f}  F1={f1*100:.2f}")

# Compute CoNLL score (average of MUC, B³, CEAF)
conll_f1 = sum(results[m]['f1'] for m in ['muc', 'bcub', 'ceafe']) / 3
print(f"\n{'CoNLL F1':10s}: {conll_f1:.2f}")

print("\n" + "="*80)
print(f"✓ Evaluation successful!")
print("="*80)
