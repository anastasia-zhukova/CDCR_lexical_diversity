# Coreference Resolution of Biased Wording Through Paraphrase Identification

Cross-document coreference resolution (CDCR) is a natural language processing task aimed at determining whether various mentions (e.g. shot, gunshot), across one or more documents, refer to the same entity or event. Similarly, paraphrase identification predicts if a pair of sentences describe the same situation **in a more general context**. This repository explores what happens when the two tasks are combined in a Multi-task learning setup and trained over news datasets to resolve coreference of mentions in media bias dataset.

## Table of Contents
1. [Installation](#1-installation)
2. [Datasets](#2-datasets)
3. [Training](#3-training)
4. [Predict Pairs](#4-predict-pairs)
5. [Predict Clusters](#5-predict-clusters)
6. [Agglomerative Clustering and Evaluation](#6-agglomerative-clustering-and-evaluation)
7. [Notes](#notes)

## 1. Installation

1. **Python 3.8 required**
2. Recommended to create a virtual environment (venv).
3. Install libraries: `pip install -r requirements.txt`

## 2. Datasets

The CDCR datasets used can be found in the [diverse CDCR datasets repository](https://github.com/anastasia-zhukova/Diverse_CDCR_datasets). We used ECBplus-prep and NewsWCL50-prep datasets. Download these folders and save them in the path `/CDCR/data/`.

For paraphrase identification, we import the MRPC dataset from the transformers dataset library and we pre-process the Headlines of War dataset as explained [here](/PI/data/HoW/README.md).

## 3. Training

We used 2 sampling strategies to train the multi-task learning model:

1. Sequential sampling: Batches contain data only from one dataset at a time. The likelihood of a batch having data of each dataset is proportional to its size.

2. Simultaneous: Each batch contains data from all the datasets in proportion to its size.

The next steps are run from the path: `/MTL/cross_encoder_adapted/`

The sampling strategy is defined in the `sampling` parameter of the `configs/config_pairwise.json` file.

We defined 4 experiments or combinations of datasets:
1. ECB+ only: Uses ECB+ entities and events
2. MRPC only: Uses MRPC only
3. ECB+ and MRPC: Combines the two benchmark datasets
4. ECB+, MRPC, and HoW: Combines the two benchmark datasets with HoW.

To change the experiment, modify the `experiment_id` parameter in the `configs/config_pairwise.json` file. The possible values range from Experiment1 to Experiment4. Then, run the code:

 ```bash
 python train.py 
 ```

 After training, the models' checkpoints are stored in the following path:
`models/pairwise_scorers/Experiment{{i}}/{{model size}}_{{batch size}}/{{training strategy}}/checkpoint_{{i}}`

## 4. Predict Pairs

Choose the model to predict by changing the `chosen_model_path` parameter in `configs/config_pairwise.json`. Then run `predict_pairwise.py` to predict pairs from the 4 datasets. The results are stored in the folder defined in `chosen_model_path`.

## 5. Predict Clusters

Move the predicted pairs to the`/results` folder. Then, run the `predict_clusters.py` scripts defining the CDCR dataset. The options are ECBplus-prep or NewsWCL50-prep.

 ```bash
 python predict_clusters.py <dataset_name>
 ```

This code exports the gold and predicted clusters in the CoNLL format following this convention:
`key_{{dataset_name}}.conll` -> Gold clusters
`sys_{{dataset_name}}_{{cluster}}_exp{{i}}_{{training_strategy_abbreviation}}_sampling.conll` -> Predicted clusters

    - {{dataset_name}} is a placeholder for the dataset name.
    - {{cluster}} is a placeholder for the cluster.
    - {{i}} is a placeholder for the experiment number or identifier.
    - {{training_strategy_abbreviation}} is a placeholder for the training strategy abbreviation.
## 6. Agglomerative Clustering and Evaluation

After creating the CoNLL files, run the `report_metrics.py` script to create a markdown file with the results. The evaluation uses a [python implementation](https://github.com/ns-moosavi/coval/) of the [official CoNLL implementation](https://github.com/conll/reference-coreference-scorers). There is no threshold tuning. We used the default value of 0.7 used in the [WEC cross-document event coreference model](https://github.com/AlonEirew/cross-doc-event-coref). 

The `result.md` file has the following structure for each file with predictions:

### File: predicted_pairs_exp3_sim_sampling.csv

| Dataset   |   Precision |   Recall |   F1 Score |
|:----------|------------:|---------:|-----------:|
| how       |    0.756685 | 0.560095 |   0.616035 |
| mrpc      |    0.54082  | 0.442899 |   0.447406 |

#### ECBplus-prep

| Metric   |   Recall |   Precision |       F1 |
|:---------|---------:|------------:|---------:|
| mentions | 100      |    100      | 100      |
| muc      |  85.1307 |     75.4799 |  80.0154 |
| bcub     |  78.8259 |     63.2776 |  70.2011 |
| ceafe    |  57.6969 |     74.7018 |  65.1073 |
| lea      |  61.9916 |     53.4232 |  57.3893 |

CoNLL score: 71.77
#### NewsWCL50-prep

| Metric   |   Recall |   Precision |        F1 |
|:---------|---------:|------------:|----------:|
| mentions | 100      |   100       | 100       |
| muc      |  72.5577 |    90.9937  |  80.7366  |
| bcub     |  40.6999 |    62.5674  |  49.3183  |
| ceafe    |  41.8759 |     4.90505 |   8.78149 |
| lea      |  38.9816 |    46.2099  |  42.2891  |

CoNLL score: 46.28

## 7. Notes

* The CDCR model is based on [the cross encoder repository](https://github.com/ariecattan/cross_encoder/) and the training strategies follow the definitions outlined in [Analyzing Multi-Task Learning for Abstractive Text Summarization](https://arxiv.org/abs/2210.14606).
