import argparse
import os
import pandas as pd
import pyhocon
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import RobertaModel
from torch.utils import data
import torch
from tqdm import tqdm

from dataset import CrossEncoderDatasetFull,CrossEncoderMixedDatasetFull, ParaphraseDataset
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
    datasets_config = resolve_data_path(datasets_config)
    config_experiments = pyhocon.ConfigFactory.parse_file(args.config_experiments)
    
    logger = create_logger(config, create_file=True)
    logger.setLevel(logging.INFO)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    #Check where I read the model to be evaluated from

    #Load data

    if "Experiment" in config.experiment_id:
        test = CrossEncoderMixedDatasetFull(config,datasets_config,config_experiments[config.experiment_id]['test'])
        
    else:
        if config.data_folder == "mrpc" or "HoW_DatasetDict" in config.data_folder:
            test = ParaphraseDataset(config, 'test') 
        else:
            test = CrossEncoderDatasetFull(config, 'test') 

    test_loader = data.DataLoader(test, batch_size=config.eval_batch_size, shuffle=False)

    #Load model 
    device_ids = config.gpu_num
    device = torch.device("cuda:{}".format(device_ids[0]))

    cross_encoder_single = FullCrossEncoder(config).to(device)
    cross_encoder_single.model = RobertaModel.from_pretrained(os.path.join(config.chosen_model_path, 'bert')).to(device)
    cross_encoder_single.linear_cdcr.load_state_dict(torch.load(os.path.join(config.chosen_model_path, 'linear_cdcr')))
    cross_encoder_single.linear_pi.load_state_dict(torch.load(os.path.join(config.chosen_model_path, 'linear_pi')))

    cross_encoder = torch.nn.DataParallel(cross_encoder_single, device_ids=device_ids)

    cross_encoder.eval()

    #Predict
    number_of_positive_pairs, number_of_pairs = 0, 0

    all_scores, all_labels, all_identifiers = [], [], []
    # Predict values

    for batch_x, batch_y,instance_id,dataset_task in tqdm(test_loader):
        
        all_identifiers.extend(instance_id)

        input_ids, attention_mask = preprocess_and_tokenize_pair(batch_x,cross_encoder_single.tokenizer,device)
        with torch.no_grad():
            scores = cross_encoder(input_ids, attention_mask,dataset_task)

        all_scores.extend(scores.detach().cpu().squeeze())
        all_labels.extend(batch_y.squeeze())
        torch.cuda.empty_cache()
        
        number_of_positive_pairs += len((batch_y == 1).nonzero())
        number_of_pairs += len(batch_y)

    all_labels_tensor = torch.stack(all_labels)
    all_scores_tensor = torch.stack(all_scores)
    strict_preds = (all_scores_tensor > 0).to(torch.int)

    subtopics_dictionary = {}
    if hasattr(test, 'mentions'):
        subtopics_dictionary = {d["mention_id"]: d['subtopic_id'] for d in test.mentions}

    subtopic_list, dataset_names_list, mention_type_list = [],[],[]

    for identifier in all_identifiers:
        identifier_sep_list = identifier.split('[SEP]')
        if len(identifier_sep_list)==2: #MRPC
            subtopic_list.append('NA')
        elif len(identifier_sep_list)==3:
            subtopic_mention_id1 = subtopics_dictionary[identifier_sep_list[1]]
            subtopic_mention_id2 = subtopics_dictionary[identifier_sep_list[2]]
            assert subtopic_mention_id1 == subtopic_mention_id2
            subtopic_list.append(subtopic_mention_id1)
            
        else:
            raise ValueError(f'Check the instance identifier, it should not have length {len(len(identifier_sep_list))}')
        
        dataset_info = extract_dataset_info_from_instance_id(identifier)
        mention_type_list.append(dataset_info['mention_type'])
        dataset_names_list.append(dataset_info['dataset_name'])

    df_score = pd.DataFrame({"instance_id":all_identifiers,
                                'dataset':dataset_names_list,
                                'mention_type':mention_type_list,
                            'subtopic':subtopic_list,
                            'label':all_labels_tensor,
                            "prediction":all_scores_tensor})
    
    df_score.to_csv(os.path.join(config.chosen_model_path, "predicted_pairs.csv"),index=False)

    # Compute precision, recall, and F1 score grouped by dataset and entity_type
    grouped_metrics = df_score.groupby(['dataset', 'mention_type']).apply(lambda x: {
        'Precision': precision_score(x['label'], x['prediction']),
        'Recall': recall_score(x['label'], x['prediction']),
        'F1 Score': f1_score(x['label'], x['prediction'])
    }).reset_index()

    grouped_metrics = pd.concat([grouped_metrics.drop(columns=0), grouped_metrics[0].apply(pd.Series)], axis=1)

    print(grouped_metrics)