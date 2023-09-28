# Coreference Resolution of Biased Wording Through Paraphrase Identification

Cross-document coreference resolution (CDCR) is a natural language processing task aimed at determining whether various mentions (e.g. shot, gunshot), across one or more documents, refer to the same entity or event. Similarly, paraphrase identification predicts if a pair of sentences describe the same situation **in a more general context**. We explore what happens when the two tasks are combined in a Multi-task learning setup and trained over news datasets to resolve coreference of mentions in media bias dataset.


## 1. Installation

1) __Python 3.8 required__
2) Recommended to create a venv.
3) Install libraries: ```pip install -r requirements.txt```

## 2. Datasets
The CDCR datasets used can be found in the [diverse cdcr datasets repository](https://github.com/anastasia-zhukova/Diverse_CDCR_datasets). We used ECBplus-prep and NewsWCL50-prep datasets. Download these folders to the path /CDCR/data/

For paraphrase identification we used the MRPC dataset imported from the transformers dataset library and a processed version of Headlines of War as explained [here](/PI/data/HoW/README.md)

## 3. Train model
We used 2 sampling strategies strategies to train the multi-task learning model.

1. Sequential sampling: batches contain data only from one dataset at the time. The likelihood of a batch having data of each dataset is proportional to its size.

2. Simultaneous: Each batch contains data from all the datasets in proportion to its size.

The sampling strategy is defined in the sampling parameter of the configs/config_pairwise.json file

We defined 4 experiments with diffent combinations of datasets:
1. ECB+ only: Uses ECB+ entities and events
2. MRPC only: Uses MRPC only
3. ECB+ and MRPC: Combines the two benchmark datasets
4. ECB+, MRPC and HoW: Combines the two benchmark datasets with HoW.

To change the experiment modify the parameter experiment_id in the configs/config_pairwise.json file. The possible values range from Experiment1 to Experiment4. After training, the models' checkpoints are stored in the following path:
models/pairwise_scorers/Experiment<i>/<model size>_<batch size>/<training strategy>/checkpoint_<i>

In this structure:

- `<i>` represents the experiment number or identifier.
- `<model size>` and `<batch size>` indicate the specific model size and batch size used.
- `<training strategy>` describes the strategy or approach used for training.
- `checkpoint_<i>` contains the checkpoint files for each experiment.


### 4. Predict pairs 
Choose the model to predict changing the parameter chosen_model_path in configs/config_pairwise.json. Then run predict_cluster.py to predict pairs from the 4 datasets. The results are stored in the /results folder

### 5. Predict clusters
Once there are predictions in the /results folder, run the predict_clusters scripts defining the CDCR dataset. The options are ECBplus-prep or NewsWCL50-prep.

 ```python predict_clusters <dataset_name>```

This code exports the gold and predicted clusters in the CoNLL format following this convention:
key_<dataset_name>.conll -> Gold clusters
sys_<dataset_name>_<cluster>_exp<i>_<training_strategy_abbreviation>_sampling.conll

### 6. Agglomerative clustering and Evaluation
After creating the conll files run the report_metrics.py script to create a markdown files with the results. The evaluation uses a [python implementation](https://github.com/ns-moosavi/coval/) of the [official CoNLL implementation](https://github.com/conll/reference-coreference-scorers). There is no threshold tuning. We used the default value of 0.7 used in the [WEC cross-document event coreference model](https://github.com/AlonEirew/cross-doc-event-coref). 
The result.md file has the following structure:
## File: predicted_pairs_exp3_sim_sampling.csv
| dataset   |   Precision |   Recall |   F1 Score |
|:----------|------------:|---------:|-----------:|
| how       |    0.756685 | 0.560095 |   0.616035 |
| mrpc      |    0.54082  | 0.442899 |   0.447406 |
### ECBplus-prep
| Metric   |   Recall |   Precision |       F1 |
|:---------|---------:|------------:|---------:|
| mentions | 100      |    100      | 100      |
| muc      |  85.1307 |     75.4799 |  80.0154 |
| bcub     |  78.8259 |     63.2776 |  70.2011 |
| ceafe    |  57.6969 |     74.7018 |  65.1073 |
| lea      |  61.9916 |     53.4232 |  57.3893 |
CoNLL score: 71.77
### NewsWCL50-prep
| Metric   |   Recall |   Precision |        F1 |
|:---------|---------:|------------:|----------:|
| mentions | 100      |   100       | 100       |
| muc      |  72.5577 |    90.9937  |  80.7366  |
| bcub     |  40.6999 |    62.5674  |  49.3183  |
| ceafe    |  41.8759 |     4.90505 |   8.78149 |
| lea      |  38.9816 |    46.2099  |  42.2891  |
CoNLL score: 46.28

### Notes

* The CDCR model is based on https://github.com/ariecattan/cross_encoder/ and the training strategies follow the definitions outlined in [Analyzing Multi-Task Learning for Abstractive Text Summarization](https://arxiv.org/abs/2210.14606)
