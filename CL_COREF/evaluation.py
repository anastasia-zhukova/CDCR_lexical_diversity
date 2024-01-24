import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F

from coval.conll.reader import get_doc_mentions, set_annotated_parse_trees, get_doc_mentions, remove_nested_coref_mentions, get_mention_assignments
from coval.eval import evaluator

###
# Merge gold and predicted clusters in one dataset
###

def return_embeddings_by_topic (dataloader,model,cl_cdcr_dataset):
    #Store vector representations by topic
    dict_embeddings_by_topic={}
    #Validation dataloader
    for indices, array_mention in dataloader: #Validation batch size should be different. It will be way faster
        # I also need to get the keys from the mentions to cluster them later
        with torch.no_grad():
            output_mentions=model(array_mention)
            batch_keys = [cl_cdcr_dataset.keys_list[i] for i in indices]

            for key, embedding in zip(batch_keys,output_mentions):
                mention_temp = cl_cdcr_dataset.cdcr_dataset.mentions[key]
                topic_number = mention_temp.topic_id

                #Stores embeddings by topic
                if topic_number in dict_embeddings_by_topic:
                    dict_embeddings_by_topic[topic_number][key] = embedding
                else:
                    dict_embeddings_by_topic[topic_number] = {key: embedding}

    return dict_embeddings_by_topic


def predict_clusters(dict_embeddings_by_topic):
    predicted_clusters_df_list = []
    max_predicted_cluster_value = 0

    for topic, dict_topic in dict_embeddings_by_topic.items():
        list_keys_dict_topic = list(dict_topic.keys())
        list_of_embeddings = [dict_topic[key] for key in list_keys_dict_topic]
        #Computes pairwise cosine similarity
        distances = torch.tensor([[1 - F.cosine_similarity(vec1, vec2, dim=0).item() for vec2 in list_of_embeddings] for vec1 in list_of_embeddings])

        #Clustering
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                                distance_threshold=0.2)
        fit = clustering.fit(distances)

        #Implement CoNLL score with this fit. I gotta merge on mention_id
        #Save as Dataframe
        adjusted_predicted_label = fit.labels_+ max_predicted_cluster_value
        predicted_df_by_topic = pd.DataFrame({"mention_id":list_keys_dict_topic,"predicted_cluster":adjusted_predicted_label,"topic_id":[topic]*len(list_keys_dict_topic)})
        predicted_clusters_df_list.append(predicted_df_by_topic)

        max_predicted_cluster_value = max(adjusted_predicted_label) +1

    predicted_clusters_df = pd.concat(predicted_clusters_df_list)
    return predicted_clusters_df

def load_gold_clusters(cl_cdcr_dataset):
    #Load gold clusters
    # Your loop to create the list of dictionaries
    mentions_data = []  

    for k, v in cl_cdcr_dataset.cdcr_dataset.mentions.items():
        mention_id = k
        topic_id = v.topic_id
        subtopic_id = v.subtopic_id  # Might be missing in NewsWCL50
        coref_chain = v.coref_chain
        mention_type = v.type

        # Create a dictionary to store the values for each mention
        mention_dict = {
            "mention_id": mention_id,
            "subtopic_id": subtopic_id,
            "coref_chain": coref_chain,
            "mention_type": mention_type
        }

        # Append the dictionary to the list
        mentions_data.append(mention_dict)

    # Create a DataFrame from the list of dictionaries
    gold_clusters_df = pd.DataFrame(mentions_data)

    set_coref_chains = gold_clusters_df.coref_chain.unique()
    enum_dict_coref = {item: index for index, item in enumerate(set_coref_chains)}
    gold_clusters_df["gold_cluster"] = gold_clusters_df["coref_chain"].map(enum_dict_coref)

    return gold_clusters_df

def return_gold_predicted_clusters(cl_cdcr_dataset,dataloader,model):
    gold_clusters_df = load_gold_clusters(cl_cdcr_dataset)
    dict_embeddings_by_topic = return_embeddings_by_topic (dataloader,model,cl_cdcr_dataset)
    predicted_clusters_df = predict_clusters(dict_embeddings_by_topic)
    merged_df = gold_clusters_df.merge(predicted_clusters_df,how="inner", on="mention_id")
    return merged_df



### 
# Compute CoNLL Score
###

def create_doc_lines(dataset_name,predictions_df,column_name):
    dict_key = f"({dataset_name}); part 000\n"
    conll_dict = {dict_key:[]}
    inner_list = []
    for _,row in predictions_df.iterrows():

        inner_list.append(f"{dataset_name}\t({row[column_name]})" + "\n")


    conll_dict[dict_key].append(inner_list)

    return conll_dict

def get_coref_infos(dataset_name, predictions_df,key_column_name,prediction_column_name, 
                    NP_only=False, remove_nested=False, keep_singletons=True, min_span=False):
    
    key_doc_lines = create_doc_lines(dataset_name=dataset_name, predictions_df=predictions_df, column_name=key_column_name)
    sys_doc_lines = create_doc_lines(dataset_name=dataset_name, predictions_df=predictions_df, column_name=prediction_column_name)
    doc_coref_infos = {}

    key_nested_coref_num = 0
    sys_nested_coref_num = 0
    key_removed_nested_clusters = 0
    sys_removed_nested_clusters = 0
    key_singletons_num = 0
    sys_singletons_num = 0

    for doc in key_doc_lines:
        key_clusters, singletons_num = get_doc_mentions(doc, key_doc_lines[doc], keep_singletons)
        key_singletons_num += singletons_num

        if NP_only or min_span:
            key_clusters = set_annotated_parse_trees(key_clusters, key_doc_lines[doc], NP_only, min_span)

        sys_clusters, singletons_num = get_doc_mentions(doc, sys_doc_lines[doc], keep_singletons)
        sys_singletons_num += singletons_num

        if NP_only or min_span:
            sys_clusters = set_annotated_parse_trees(sys_clusters, key_doc_lines[doc], NP_only, min_span)

        if remove_nested:
            nested_mentions, removed_clusters = remove_nested_coref_mentions(key_clusters, keep_singletons)
            key_nested_coref_num += nested_mentions
            key_removed_nested_clusters += removed_clusters

            nested_mentions, removed_clusters = remove_nested_coref_mentions(sys_clusters, keep_singletons)
            sys_nested_coref_num += nested_mentions
            sys_removed_nested_clusters += removed_clusters

        sys_mention_key_cluster = get_mention_assignments(sys_clusters, key_clusters)
        key_mention_sys_cluster = get_mention_assignments(key_clusters, sys_clusters)

        doc_coref_infos[doc] = (key_clusters, sys_clusters, key_mention_sys_cluster, sys_mention_key_cluster)

    if remove_nested:
        print('Number of removed nested coreferring mentions in the key annotation: %s; and system annotation: %s' % (
            key_nested_coref_num, sys_nested_coref_num))
        print('Number of resulting singleton clusters in the key annotation: %s; and system annotation: %s' % (
            key_removed_nested_clusters, sys_removed_nested_clusters))

    if not keep_singletons:
        print('%d and %d singletons are removed from the key and system files, respectively' % (
            key_singletons_num, sys_singletons_num))

    return doc_coref_infos


def compute_metrics_cdcr(doc_coref_infos, metrics):

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
    
    df_metrics = pd.DataFrame(evaluation_results).set_index("Metric")
    conll_f1 = df_metrics.loc[["muc", "bcub", "ceafe"], "F1"].mean()
    df_metrics.loc["conll", "F1"] = conll_f1

    return df_metrics


# Params
allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
        ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
        ('lea', evaluator.lea)]


def compute_cdcr_metrics(cl_cdcr_dataset,dataloader,model,metrics,NP_only=False, 
                         remove_nested=False, keep_singletons=True, min_span=False):
    #Define metrics
    if 'all' in metrics:
        metrics = allmetrics
    else:
        metrics = [(name, metric) for name, metric in allmetrics
                if name in metrics]
        if not metrics:
            metrics = allmetrics

    #Compute metrics
    merged_df = return_gold_predicted_clusters(cl_cdcr_dataset,dataloader,model)

    doc_coref_infos = get_coref_infos(dataset_name=cl_cdcr_dataset.cdcr_dataset.name, predictions_df=merged_df,
                key_column_name='gold_cluster',prediction_column_name='predicted_cluster', 
                    NP_only=NP_only, remove_nested=remove_nested, keep_singletons=keep_singletons, min_span=min_span)
    
    df_metrics = compute_metrics_cdcr(doc_coref_infos=doc_coref_infos,metrics=metrics)

    return df_metrics, merged_df





# Old code reading from CoNLL files
# import os

# def format_line(dataset_name,row, cluster):
#     return f"{dataset_name}\t({row[cluster]})"


# def write_output_to_file(df, dataset_name, output_file_name, cluster_col_name):
#     # Construct the output directory path
#     output_directory_path = os.path.join(os.getcwd(), "conll_files")

#     # Ensure the output directory exists; if not, create it
#     if not os.path.exists(output_directory_path):
#         os.makedirs(output_directory_path)

#     output_file = os.path.join(output_directory_path, output_file_name)
#     with open(output_file, "w") as f:
#         f.write(f"#begin document ({dataset_name}); part 000\n")
#         for _, row in df.iterrows():
#             f.write(format_line(dataset_name, row, cluster_col_name) + "\n")
#         f.write("#end document\n")

#     print(f"File '{output_file}' written successfully.")

# key_filename=f"key_{cl_cdcr_val.cdcr_dataset.name}.conll"
# write_output_to_file(merged_df,cl_cdcr_val.cdcr_dataset.name,key_filename,"gold_cluster")
# sys_filename=f"sys_{cl_cdcr_val.cdcr_dataset.name}.conll"
# write_output_to_file(merged_df,cl_cdcr_val.cdcr_dataset.name,sys_filename,"predicted_cluster")

# #Report metrics
# computed_metrics = compute_metrics_cdcr(os.path.join(os.getcwd(), "conll_files",key_filename),
#                      os.path.join(os.getcwd(), "conll_files",sys_filename),
#                        metrics, NP_only, remove_nested,
#         keep_singletons, min_span)

# computed_metrics = computed_metrics.set_index("Metric")
# computed_metrics

# conll=computed_metrics.loc["muc", "F1"] + computed_metrics.loc["bcub", "F1"] + computed_metrics.loc["ceafe", "F1"]
# conll = (conll / 3)

# conll