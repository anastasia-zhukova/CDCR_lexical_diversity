import os
import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from coval.conll import reader
from coval.conll import util
from coval.eval import evaluator
from utils import load_results_to_dict

def compute_metrics_cdcr(key_file, sys_file, metrics, NP_only, remove_nested,
        keep_singletons, min_span):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,
            remove_nested, keep_singletons, min_span)

    evaluation_results = []

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)
        
        evaluation_results.append({
            'Metric': name,
            'Recall': recall * 100,
            'Precision': precision * 100,
            'F1': f1 * 100
        })

    return pd.DataFrame(evaluation_results)

def compute_metrics_pi(df):
    # Filter the DataFrame
    df_filtered = df.query("dataset=='mrpc' or dataset=='how'").copy()
    df_filtered['prediction'] = (df_filtered.prediction > 0).astype(int)

    # Calculate precision, recall, F1 score, and support
    grouped_metrics = df_filtered.groupby(['dataset']).apply(lambda x: pd.Series(
        precision_recall_fscore_support(x['label'], x['prediction'], average='weighted')[:-1],  # Drop the 'support' column
        index=['Precision', 'Recall', 'F1 Score']
    ))
    
    return grouped_metrics

def main():
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)]


    sys_folder = "conll_files/"  # Folder containing sys files

    NP_only = 'NP_only' in sys.argv
    remove_nested = 'remove_nested' in sys.argv
    keep_singletons = ('remove_singletons' not in sys.argv
            and 'removIe_singleton' not in sys.argv)
    min_span = False
    if ('min_span' in sys.argv 
        or 'min_spans' in sys.argv
        or 'min' in sys.argv):
        min_span = True
        # has_gold_parse = util.check_gold_parse_annotation(key_file)
        # if not has_gold_parse:
        #         util.parse_key_file(key_file)
        #         key_file = key_file + ".parsed"

    if 'all' in sys.argv:
        metrics = allmetrics
    else:
        metrics = [(name, metric) for name, metric in allmetrics
                if name in sys.argv]
        if not metrics:
            metrics = allmetrics

    # List sys files in the sys_folder that start with "sys"
    sys_files = [f for f in os.listdir(sys_folder) if f.startswith("sys")]

   
    #Load predictions
    dict_predictions = load_results_to_dict("results/")

        # Open a Markdown file for writing
    with open('results.md', 'w') as markdown_file:

        # Iterate through dictionary items
        for k, v in dict_predictions.items():
            model_id  = k.split(".")[0].split('_', 2)[-1]

            # Write file name to Markdown
            markdown_file.write(f"## File: {k}\n")
            
            # Write PI metrics table to Markdown
            metrics_pi = compute_metrics_pi(v)
            markdown_file.write(metrics_pi.to_markdown() + '\n')
            # Write CDCE metrics to markdown
            for dataset in ["ECBplus-prep","NewsWCL50-prep"]:
                sys_file = [sysfile for sysfile in sys_files if model_id in sysfile and dataset in sysfile][0]
                sys_file_path = os.path.join(sys_folder, sys_file)
                key_file = f"conll_files/key_{dataset}.conll"  # Use the key file here
                results_df = compute_metrics_cdcr(key_file, sys_file_path, metrics, NP_only, remove_nested,
                    keep_singletons, min_span)
                results_df.set_index("Metric", inplace=True)

                markdown_file.write(f"### {dataset}\n")
                markdown_file.write(results_df.to_markdown() + '\n')
                
                metric_names = {item[0] for item in metrics}
                metrics_for_conll = {'muc', 'bcub', 'ceafe'}
                if metrics_for_conll.issubset(metric_names):
                    conll=results_df.loc["muc", "F1"] + results_df.loc["bcub", "F1"] + results_df.loc["ceafe", "F1"]
                    conll = (conll / 3)
                    markdown_file.write('CoNLL score: {:.2f}'.format(conll))
                    markdown_file.write("\n")



if __name__ == '__main__':
    main()