import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F

from coval.conll.reader import get_doc_mentions, set_annotated_parse_trees, remove_nested_coref_mentions, get_mention_assignments
from coval.eval import evaluator

# Params
allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
        ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
        ('lea', evaluator.lea),('pd_lea',evaluator.pd_lea)]

def create_embeddings_by_mention(dataloader,device,model,cl_dataset_keys_list):
    dict_embeddings_by_mention= {}

    for indices, input_ids, attention_masks in dataloader:
        with torch.no_grad():
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            embeddings = model(input_ids,attention_masks)
        
        keys_list = [cl_dataset_keys_list[index] for index in indices.tolist()]

        dict_embeddings_batch = {key: emb for key, emb in zip(keys_list, embeddings)}
        # Update the main dictionary
        dict_embeddings_by_mention.update(dict_embeddings_batch)
    return dict_embeddings_by_mention

def return_embeddings_by_topic(dict_embeddings_by_mention,mentions):
    dict_embeddings_by_topic={}

    for mention_key,embedding in dict_embeddings_by_mention.items():
        topic_id_temp = mentions[mention_key].topic_id
        if topic_id_temp in dict_embeddings_by_topic:
            dict_embeddings_by_topic[topic_id_temp][mention_key] = embedding
        else:
            dict_embeddings_by_topic[topic_id_temp] = {mention_key: embedding}
    return dict_embeddings_by_topic

def predict_clusters(dict_embeddings_by_topic,clustering_threshold=0.2):
    predicted_clusters_df_list = []
    max_predicted_cluster_value = 0

    for topic, dict_topic in dict_embeddings_by_topic.items():
        list_of_embeddings = list(dict_topic.values())
        #Computes pairwise cosine similarity
        distances = torch.tensor([[1 - F.cosine_similarity(vec1, vec2, dim=0).item() for vec2 in list_of_embeddings] for vec1 in list_of_embeddings])

        #Clustering
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                                distance_threshold=clustering_threshold)
        #Implement CoNLL score with this fit and save as Dataframe
        adjusted_predicted_label = clustering.fit_predict(distances) + max_predicted_cluster_value #Ensure no label overlap when moving to another topic
        predicted_df_by_topic = pd.DataFrame({"mention_id":list(dict_topic.keys()),
                                              "predicted_cluster":adjusted_predicted_label,"topic_id":[topic]*len(dict_topic)})
        predicted_clusters_df_list.append(predicted_df_by_topic)

        max_predicted_cluster_value = max(adjusted_predicted_label) +1

    predicted_clusters_df = pd.concat(predicted_clusters_df_list)
    return predicted_clusters_df

### 
# Compute CoNLL Score
###

def create_doc_lines(dataset_name,predictions_df,column_name):
    dict_key = f"({dataset_name}); part 000\n"
    conll_dict = {dict_key:[]}
    inner_list = []
    for _,row in predictions_df.iterrows():

        inner_list.append(f"{dataset_name}\t{row['mention_id']}\t{row['coref_chain']}\t({row[column_name]})" + "\n")


    conll_dict[dict_key].append(inner_list)

    return conll_dict

def get_coref_infos(dataset_name, predictions_df,
                    NP_only=False, remove_nested=False, keep_singletons=True, min_span=False):
    
    key_column_name='gold_cluster'
    prediction_column_name='predicted_cluster'

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

def compute_cdcr_metrics(cdcr_dataset_name,predictions_df,mention_type,metrics="all",NP_only=False, 
                         remove_nested=False, without_singletons=False, min_span=False,):
    
    keep_singletons = True
    if without_singletons:
        keep_singletons=False
    #Define metrics 
    metrics = allmetrics if 'all' in metrics else [(name, metric) for name, metric in allmetrics if name in metrics]

    #Compute metrics
    doc_coref_infos = get_coref_infos(dataset_name=cdcr_dataset_name, predictions_df=predictions_df,
                    NP_only=NP_only, remove_nested=remove_nested, keep_singletons=keep_singletons, min_span=min_span)
    #Here I join the head words info extracting mention index from each mention, and pass it onto evaluator

    dict_phrasing_diversity = dict(zip(predictions_df.mention_id, zip(predictions_df.mention_head,predictions_df.mention_wordset,predictions_df.mention_wo_stopwords)))

    evaluation_results = []
    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,dict_phrasing_diversity,metric,dataset_name=cdcr_dataset_name,mention_type= mention_type,beta=1)
        
        evaluation_results.append({
            'Metric': name,
            'Recall': recall * 100,
            'Precision': precision * 100,
            'F1': f1 * 100
        })

    df_metrics = pd.DataFrame(evaluation_results).set_index("Metric")
    df_metrics.loc["conll", "F1"] = df_metrics.loc[["muc", "bcub", "ceafe"], "F1"].mean()
    return df_metrics

def predict_and_evaluate(dataloader,device,model,cl_cdcr_dataset,metrics="all",without_singletons=False,clustering_threshold=0.2):
    cdcr_dataset=cl_cdcr_dataset.cdcr_dataset
    dict_embeddings_by_mention = create_embeddings_by_mention(dataloader=dataloader,device=device,model=model,cl_dataset_keys_list=cl_cdcr_dataset.keys_list)
    dict_embeddings_by_topic=return_embeddings_by_topic(dict_embeddings_by_mention=dict_embeddings_by_mention,mentions=cdcr_dataset.mentions)
    predicted_clusters_df = predict_clusters(dict_embeddings_by_topic,clustering_threshold)
    merged_df = cdcr_dataset.gold_clusters_df.merge(predicted_clusters_df,how="inner", on="mention_id")
    df_metrics = compute_cdcr_metrics(cdcr_dataset_name=cdcr_dataset.name,predictions_df=merged_df,metrics=metrics,without_singletons=without_singletons,mention_type=cl_cdcr_dataset.mention_type)
    print(f' eval conll: {round(df_metrics.loc["conll","F1"],2)},eval lea: {round(df_metrics.loc["lea","F1"],2)},eval pd_lea: {round(df_metrics.loc["pd_lea","F1"],2)}')
    return df_metrics, merged_df