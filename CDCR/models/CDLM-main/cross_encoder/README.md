# Cross-Document Coreference Resolution
We heavily relied on the code from the [cross-encoder cross-document coreference resolution](https://github.com/ariecattan/cross_encoder) repository.
The instructions are based on the [cross-document coreference resolution](https://github.com/ariecattan/coref) repository, and are modified to support our version.
## Getting started - packages and data
* INSTALL WITH PYTHON 3.7!!
* Install python3 requirements `pip install -r requirements.txt`.
* Run `python -m spacy download en_core_web_sm`.
* Download the ECB+ corpus from [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).
* THE REPO CONTAINS THE ECB+ FOLDER ZIPPED. IT IS NECESSARY TO UNZIP IT FIRST.
* AFTER UNZIPPING, TWO FOLDERS ARE EXTRACTED: ECB+ AND _MACOSX (I SUPPOSE DEPENDING ON THE OS)
THE ECB+ HAS TO MOVE UP ONE LEVEL OVERWRITTING THE FOLDER CONTAININT IT

* Run the following script in order to extract the data from ECB+ dataset.
 and build the gold conll files: 
```python
python get_ecb_data.py --data_path path_to_data

Currently:
python get_ecb_data.py --data_path /home/sago/Documents/GitHub/ecbPlus/ECB+_LREC2014
```



## Training Instructions


The core of our model is the pairwise scorer between two spans, 
estimating the probability of how two spans belong to the same cluster.

 
#### Mention Types

In ECB+, the entity and event coreference clusters are annotated separately, 
making it possible to train a model only on event or entity coreference. 
Therefore, you can train events, entity, or both (in our paper we reported the results of the separate modes).
You can set the value of the `mention_type` in the config file (under the configs directory) 
to `events`, `entities` or `mixed`.



#### Running the model
 
```python
python train.py
```

Note: Make sure to place the tokenizer files inside the model directory.
## Prediction

Given the trained pairwise scorer, we use an agglomerative
clustering procedure in order to cluster the candidate spans into coreference clusters. 


```python
python predict_long.py --config configs/config_pairwise_long_reg_span.json
```

(`model_path` corresponds to the directory in which you've stored the trained models)

## Evaluation

### Validation Phase 
The output of the `predict_long.py` script is a file in the standard conll format. 
Then, it's straightforward to evaluate it with its corresponding 
gold conll file (created in the first step). To that end, use the `run_scorer_long.py` script.
### Test Phase 
Given the results of the validation phase (best epoch and agglomerative clustering mode), run the `predict_long_test.py`
 and it's corresponding scorer script, ```python
  run_scorer_long_test.py <path> <mention_type>```
, using the path of the produced 
 conll file and the mention type, to obtain the final results on the test set.