This seems to be the building block of CDCR encoding:
https://github.com/ariecattan/cross_encoder
So I will analyze it and replicate it in depth

Conll.py: Overall, this code processes predictions, document IDs, start/end indices, and token data to generate CoNLL output files in both corpus level and topic level formats.

get_dict_map(predictions, doc_ids, starts,ends) Overall structures predictions to ConLL
takes a dictionary of predictions whose key is the cluster id, and the value is the list of mentions belonging to that cluster. The list of mentions are indices.
- Each mention is stored with a key containing: doc_id + index of the first letter of the mention in the sequence, I suppose, and the value is the cluster_id

------------------

dataset.py

This is the code I need to explore to understand the encoding well.

Overall this code prepares this data in the encoded format:

 [tara] [Reid] [has] [START] [entered] [END] [a] [rehab] [program] [SEP] [another] [star] [START] [into] [END] [due] [to] [excess] [SEP]     
 And combines all possible mentions within each topic, creating a label tensor of 1s if the coref ID is the same.

 ----------------------
 evaluator.py
 
 Computes classification metrics
 