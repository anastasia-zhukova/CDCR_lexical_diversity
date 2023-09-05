import argparse
import numpy as np
import os
import pandas as pd
import pyhocon
from sklearn.cluster import AgglomerativeClustering
from transformers import RobertaModel
from torch.utils import data
import torch
from tqdm import tqdm

from dataset import CrossEncoderDatasetFull,CrossEncoderMixedDatasetFull
from cluster_documents import return_mentions_with_predicted_topics
from models import FullCrossEncoder
from utils import *



if __name__ == '__main__':

    #Read config_paiwise.json file and config_datasets.json file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise.json')
    parser.add_argument('--dataset_config', type=str, default='configs/config_datasets.json')
    parser.add_argument('--config_experiments', type=str, default='configs/config_experiments.json')
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.parse_file(args.config)
    datasets_config = pyhocon.ConfigFactory.parse_file(args.dataset_config)
    config_experiments = pyhocon.ConfigFactory.parse_file(args.config_experiments)

    #Set data paths according to the common_data_folder path in the config_datasets file
    common_data_folder = datasets_config.get_string("common_data_folder")
    for key in datasets_config:
        if isinstance(datasets_config[key], dict) and "data_folder" in datasets_config[key]:
            datasets_config[key]["data_folder"] = datasets_config[key]["data_folder"].replace("${common_data_folder}", common_data_folder)

    

    config = pyhocon.ConfigFactory.parse_file(args.config)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    #Load data
    # init train and dev set
    if "Experiment" in config.experiment_id:
        test = CrossEncoderMixedDatasetFull(config,datasets_config,config_experiments[config.experiment_id]['test'])
        
    else:
        test = CrossEncoderDatasetFull(config, 'test')

        predict_topics = True #TODO Pass by config parameter
        if predict_topics:
            mentions_w_predicted_topics = return_mentions_with_predicted_topics(test)
            test.assign_mentions_with_new_topics(mentions_w_predicted_topics)

        test.create_mention_pairs_by_topics(use_predicted=True)
        test.prepare_pair_of_mentions()


    test_loader = data.DataLoader(test, batch_size=config.batch_size, shuffle=False)

    #Load model 
    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    cross_encoder_single = FullCrossEncoder(config).to(device)
    cross_encoder_single.model = RobertaModel.from_pretrained(os.path.join(config.model_path, 'bert')).to(device) # I have to choose the right model path
    cross_encoder_single.linear.load_state_dict(torch.load(os.path.join(config.model_path, 'linear')))
    cross_encoder = torch.nn.DataParallel(cross_encoder_single, device_ids=device_ids)
    cross_encoder.eval()

  

    #Set agglomerative clustering
    clustering_5 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.5)
    clustering_55 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.55)
    clustering_6 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.6)
    clustering_65 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.65)
    clustering_7 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.7)

    clustering = [clustering_5, clustering_55, clustering_6, clustering_65, clustering_7]
    clusters = [list(), list(), list(), list(), list()]
    max_ids = [0, 0, 0, 0, 0]
    threshold = {id: thresh for id, thresh in enumerate([0.5, 0.55, 0.6, 0.65, 0.7])}

    #Predict
    number_of_positive_pairs, number_of_pairs = 0, 0

    all_scores, all_labels = [], []
    # Predict values

    for batch_x, batch_y,instance_id,dataset_task in tqdm(test_loader):
        bert_tokens = cross_encoder.module.tokenizer(batch_x, pad_to_max_length=True)
        input_ids = torch.tensor(bert_tokens['input_ids'], device=device)
        attention_mask = torch.tensor(bert_tokens['attention_mask'], device=device)

        with torch.no_grad():
            # scores = cross_encoder(input_ids, attention_mask)
            # Replace temporarily to get some scores
            scores = cross_encoder(input_ids, attention_mask)
            scores = torch.sigmoid(scores)
            all_scores.extend(scores.detach().cpu().squeeze(1))
        all_labels.extend(batch_y.squeeze(1))
        torch.cuda.empty_cache()
        
        number_of_positive_pairs += len((batch_y == 1).nonzero())
        number_of_pairs += len(batch_y)

    all_labels_tensor = torch.stack(all_labels)
    all_scores_tensor = torch.stack(all_scores)

    mentions_dictionary = {d["mention_id"]: d for d in test.mentions}
    df_score = pd.DataFrame({"mention1":[id.split("[SEP]")[1] for id in test.instance_ids],
                "mention2":[id.split("[SEP]")[2] for id in test.instance_ids],
                "similarity_score":torch.rand(len(test.instance_ids))})
    
    df_score["predicted_topic1"] = df_score["mention1"].map(mentions_dictionary).apply(lambda x: x.get("predicted_topic") if x else None)
    df_score["predicted_topic2"] = df_score["mention2"].map(mentions_dictionary).apply(lambda x: x.get("predicted_topic") if x else None)
    df_score.to_csv("predicted_clusters.csv")