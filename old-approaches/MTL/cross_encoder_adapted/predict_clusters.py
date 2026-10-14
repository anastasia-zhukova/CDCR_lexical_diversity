from enum import Enum
import numpy as np
import os
import pandas as pd
from pyhocon import ConfigFactory
from sklearn.cluster import AgglomerativeClustering
import sys

from dataset import CrossEncoderDatasetFull
from utils import load_results_to_dict


##################
#### Load gold clusters
##################

# Define constants
ENTITY_MENTIONS = "entity_mentions"
EVENT_MENTIONS = "event_mentions"
SPLIT_NAMES = ['train', 'val', 'test']

# Check if the user provided a dataset name as an argument
if len(sys.argv) < 2:
    print("Usage: python script.py <dataset_name>")
    sys.exit(1)

# Get the dataset name from the command-line argument
DATASET = sys.argv[1]

# Check if the provided dataset name is valid
allowed_datasets = ["ECBplus-prep", "NewsWCL50-prep"]
if DATASET not in allowed_datasets:
    print(f"Error: Dataset must be one of the following: {', '.join(allowed_datasets)}")
    sys.exit(1)

DATA_FOLDER = f"../../CDCR/data/{DATASET}/output_data/"

class EntityType(Enum):
    ENTITIES = "entities"
    EVENTS = "events"

# Read Config
config = ConfigFactory.parse_file('configs/config_pairwise.json')
config.put('data_folder', DATA_FOLDER)

def create_datasets(config, split_name):
    datasets = {}
    for mention_type, ent_type in ((ENTITY_MENTIONS, EntityType.ENTITIES), (EVENT_MENTIONS, EntityType.EVENTS)):
        config.put('mention_type', mention_type)
        datasets[ent_type] = CrossEncoderDatasetFull(config, split_name=split_name, negative_sampling=True)
    return datasets

datasets = {}
for split_name in SPLIT_NAMES:
    datasets[split_name] = create_datasets(config.copy(), split_name)

list_dfs_mentions = []

for split_name in SPLIT_NAMES:
    for ent_type, dataset in datasets[split_name].items():
        mention_ids = [mention['mention_id'] for mention in dataset.mentions]
        subtopic_ids = [mention['subtopic_id'] for mention in dataset.mentions]
        coref_chains = [mention['coref_chain'] for mention in dataset.mentions]
        
        df_temp = pd.DataFrame({
            'mention_id': mention_ids,
            'subtopic': subtopic_ids,
            'coref_chain': coref_chains,
            'mention_type': [ent_type.value] * len(mention_ids)
        })
        list_dfs_mentions.append(df_temp)

df_all_mentions = pd.concat(list_dfs_mentions)

###################
##### Load predicted pairs
###################

def extract_mention_i(text,i):
    text_list = text.split("[SEP]")
    return text_list[i]

def predict_clusters(input_df):
    df_exp = input_df.copy()

    #Extract mentions id
    df_exp["mention1_id"] = df_exp.instance_id.apply(lambda x:extract_mention_i(x,1) )
    df_exp["mention2_id"] = df_exp.instance_id.apply(lambda x:extract_mention_i(x,2) )

    # 1. Turn logit into probability
    df_exp['predicted_prob'] = 1 / (1 + np.exp(-df_exp['prediction']))
    # 2. Compute distance
    df_exp['distance'] = 1-df_exp['predicted_prob'] 
    # 3. Map mentions to indices

    first_set = set(df_exp.mention1_id)
    second_set = set(df_exp.mention2_id)
    union_set = first_set.union(second_set)
    num_mentions = len(union_set)

    mention_index_dict = {value: idx for idx, value in enumerate(union_set)}
    index_mention_dict = {idx: value for idx, value in enumerate(union_set)}

    df_exp["mention1_id_index"] =  df_exp["mention1_id"].map(mention_index_dict)
    df_exp["mention2_id_index"] =  df_exp["mention2_id"].map(mention_index_dict)
    # Create square matrix of 0s (minimum distance) and replace combinations with actual distances
    distance_matrix = np.zeros((num_mentions,num_mentions))

    mention1_indices = df_exp["mention1_id_index"].values
    mention2_indices = df_exp["mention2_id_index"].values
    distances = df_exp["distance"].values

    distance_matrix[mention1_indices, mention2_indices] = distances
    distance_matrix[mention2_indices, mention1_indices] = distances

    #Name matrix with mention ids
    final_distance_matrix = pd.DataFrame(distance_matrix)

    column_names = []
    for col in final_distance_matrix.columns:
        column_names.append(index_mention_dict[col])

    final_distance_matrix.columns=column_names

    #Apply agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                            distance_threshold=0.7)
    fit = clustering.fit(final_distance_matrix)
    return pd.DataFrame({'mention_id':fit.feature_names_in_,"predicted_cluster":fit.labels_})

def process_predictions(dict_predictions, dataset_name):
    dict_predicted_clusters = {}
    
    for file_name, pairwise_predictions in dict_predictions.items():
        list_predicted_clusters_per_file = []
        chosen_df = pairwise_predictions.query(f"dataset == '{dataset_name}'")
        unique_mention_types = chosen_df.mention_type.unique()
        dict_combine_topics = {}
        
        new_file_key = file_name.split(".")[0].replace('predicted_pairs', 'cluster')
        
        for mention_type in unique_mention_types:
            dict_combine_topics[mention_type] = chosen_df[chosen_df['mention_type'] == mention_type].subtopic.unique()
        
        for entity_type, subtopics in dict_combine_topics.items():
            for subtopic in subtopics:
                chosen_df_subtopic = chosen_df[(chosen_df['mention_type'] == entity_type) & (chosen_df['subtopic'] == subtopic)]
                predicted_clusters_df = predict_clusters(chosen_df_subtopic)
                predicted_clusters_df = predicted_clusters_df.rename(columns={'predicted_cluster': new_file_key})
                list_predicted_clusters_per_file.append(predicted_clusters_df)
        
        dict_predicted_clusters[new_file_key] = pd.concat(list_predicted_clusters_per_file)
    
    concat_df = list(dict_predicted_clusters.values())[0]
    for _, df in list(dict_predicted_clusters.items())[1:]:
        concat_df = concat_df.merge(df, on='mention_id', how='outer')

    return concat_df

def process_cluster_columns(df):
    # Filter columns that contain the word 'cluster'
    cluster_column_names = [col for col in df.columns if 'cluster' in col]

    # Turn coreference chains into strings to feed the conll file only with integers
    set_coref_chains = df.coref_chain.unique()
    enum_dict_comp = {item: index for index, item in enumerate(set_coref_chains)}
    df["mapped_coref_chain"] = df["coref_chain"].map(enum_dict_comp)

    # Initialize max values for each cluster column
    cluster_max_values = {col: 0 for col in cluster_column_names}

    unique_subtopics_keys = df[['subtopic', 'mention_type']].drop_duplicates()
    for _, row in unique_subtopics_keys.iterrows():
        df_temp = df[(df['subtopic'] == row['subtopic']) & (df['mention_type'] == row['mention_type'])]

        # Update each cluster column for the current subtopic and mention_type
        for col in cluster_column_names:
            df_temp.loc[:, col] += cluster_max_values[col]
            df.loc[df_temp.index, col] += cluster_max_values[col]

            cluster_max_values[col] = max(df_temp[col])

    # Now cluster_max_values contains the maximum values for each cluster column
    return df, cluster_column_names

def format_line(row, cluster):
    return f"{DATASET}\t({row[cluster]})"

def write_output_to_file(df, output_path,df_column="mapped_coref_chain"):
    # Check if the directory of the output path exists; if not, create it
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(output_path, "w") as f:
        f.write(f"#begin document ({DATASET}); part 000\n")
        for _, row in df.iterrows():
            f.write(format_line(row,df_column ) + "\n")
        f.write("#end document\n")

    print(f"File '{output_path}' written successfully.")

def write_cluster_output(df, cluster_columns, output_directory):
    for cluster in cluster_columns:
        output_path = os.path.join(output_directory, f"sys_{DATASET}_{cluster}.conll")
        write_output_to_file(df, output_path,cluster)


dict_predictions = load_results_to_dict("results/")
predicted_clusters = process_predictions(dict_predictions, DATASET)

merged_values = predicted_clusters.merge(df_all_mentions,on='mention_id',how='inner')
merged_values, cluster_columns = process_cluster_columns(merged_values)

output_directory="conll_files"
write_output_to_file(merged_values, os.path.join(output_directory, f"key_{DATASET}.conll"))
write_cluster_output(merged_values, cluster_columns, output_directory)