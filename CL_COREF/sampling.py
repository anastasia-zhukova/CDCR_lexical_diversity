from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import statistics
import torch
from typing import Set, Dict

model = SentenceTransformer('all-mpnet-base-v2')

def encode_sentences(dict_sentences):

    sentences = list(dict_sentences.values())
    list_of_keys = list(dict_sentences.keys())

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)
    # Get the indices of the lower triangular elements of the similarity matrix
    lower_triangle_positions = torch.tril_indices(cosine_scores.size(0),cosine_scores.size(1), offset=-1)
    indices = lower_triangle_positions.t()
    return list_of_keys,cosine_scores, indices

def sentences_processing(dict_sentences_by_topic,mentions_by_sentence,mentions_by_type: Set[str]):
    dict_sentences_pair_similarity_by_topic = {}
    dict_mentions_by_topic_in_type = {}
    dict_mention_pairs_by_pair = {}
    for topic,sentences_dict in dict_sentences_by_topic.items():
        print(f'current topic {topic}')
        if not sentences_dict:
            print(f"topic {topic} is empty")
            continue
        
        #Compute similarity        
        list_of_keys,cosine_scores, indices = encode_sentences(sentences_dict)

        # Create a dictionary using the indices
        lower_triangle_dict = {}
        topic_mentions_set = set()
        for i, j in indices:
            key_sentence1 = list_of_keys[i.item()]
            key_sentence2 = list_of_keys[j.item()]
            #Ignore sentences without mentions of the chosen type
            if not mentions_by_sentence[key_sentence1] or not mentions_by_sentence[key_sentence2]:
                continue

            mentions_by_type_sentence1 = mentions_by_type.intersection(mentions_by_sentence[key_sentence1]) 
            mentions_by_type_sentence2 = mentions_by_type.intersection(mentions_by_sentence[key_sentence2]) 

            if not mentions_by_type_sentence1 or not mentions_by_type_sentence2:
                continue
            mentions_by_type_in_pair = mentions_by_type_sentence1.union(mentions_by_type_sentence2)
            if len(mentions_by_type_in_pair)<2:
                raise Exception(f"pair {key_sentence1},{key_sentence2} has less than 2 mentions in the current type")
            topic_mentions_set.update(mentions_by_type_in_pair)

            key_pair = (key_sentence1, key_sentence2) 
            similarity_score = round(cosine_scores[i, j].item(), 3)
            if similarity_score<1: #ignores those sentences with similarity=1. Those are mostly exact citations
                lower_triangle_dict[key_pair] = similarity_score
                dict_mention_pairs_by_pair[key_pair] = product(mentions_by_type_sentence1, mentions_by_type_sentence2)

        if lower_triangle_dict:
            dict_sentences_pair_similarity_by_topic[topic] = lower_triangle_dict
            dict_mentions_by_topic_in_type[topic] = topic_mentions_set

    return dict_sentences_pair_similarity_by_topic, dict_mentions_by_topic_in_type,dict_mention_pairs_by_pair 

def unnest_dictionary(input_dict,key=''):
  
    unnested_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            unnested_dict.update(unnest_dictionary(value, key))
        else:
            unnested_dict[key] = value

    return unnested_dict

def create_mention_pairs(dict_sentences_pairs_similarity,mentions, mention_pairs_by_pair,mention_keys_by_topic_in_type, additional_sampling = False)->dict:

    dict_positive_pairs_by_topic = defaultdict(dict)
    dict_negative_pairs_by_topic = defaultdict(dict)

    temp_dataset = []
    for topic, sentence_pairs in dict_sentences_pairs_similarity.items():
        #Compute similarity of all the mentions of the same topic    
        # dict_mention_tokens = {key:mention.tokens_str for key,mention in mentions.items() if key in mention_keys_by_topic_in_type[topic]}
        # list_of_keys,cosine_scores, _ = encode_sentences(dict_mention_tokens)
   
        topic_sentence_similarity_median = statistics.median(list(sentence_pairs.values()))
        for (sentence_key1, sentence_key2), sentence_similarity_score in sentence_pairs.items():
            key_pair = (sentence_key1, sentence_key2) 
            for mention_key1, mention_key2 in mention_pairs_by_pair[key_pair]:

                # index_key_mention1 = list_of_keys.index(mention_key1)
                # index_key_mention2 = list_of_keys.index(mention_key2)
                # mention_pair_score = round(cosine_scores[index_key_mention1, index_key_mention2].item(), 3)
                same_coref_chain = mentions[mention_key1].coref_chain == mentions[mention_key2].coref_chain
                new_dict_entry = {(mention_key1,mention_key2):sentence_similarity_score} #I can have sentence similarity score and mention similarity score.

                if same_coref_chain:
                    dict_positive_pairs_by_topic[topic].update(new_dict_entry)
                    # temp_dict_entry = {"topic":topic,"sentence1":sentence_key1,"sentence2":sentence_key2,"sentence_similarity":sentence_similarity_score,
                    #                    "mention1":mention_key1,"mention2":mention_key2,"mention_similarity":mention_pair_score,"coref":1}
                    #add to temp dataset
                else:
                    dict_negative_pairs_by_topic[topic].update(new_dict_entry)
                    # temp_dict_entry = {"topic":topic,"sentence1":sentence_key1,"sentence2":sentence_key2,"sentence_similarity":sentence_similarity_score,
                    # "mention1":mention_key1,"mention2":mention_key2,"mention_similarity":mention_pair_score,"coref":0}
                
                #temp_dataset.append(temp_dict_entry)

            
        ## Subsampling
        positive_pairs_scores= list(dict_positive_pairs_by_topic[topic].values())

        #Downsample negative pairs based on positive pairs score's Q3
        third_quartile = np.percentile(positive_pairs_scores, 75)
        dict_negative_pairs_by_topic[topic] = {key: value for key, value in dict_negative_pairs_by_topic[topic].items() if value > third_quartile}
        
        #Downsample positive pairs based on topic's median
        dict_positive_pairs_by_topic[topic] = {key: value for key, value in dict_positive_pairs_by_topic[topic].items() if value < topic_sentence_similarity_median}

        #Second subsampling based on mention's number (6sqrt(n))
        if additional_sampling:
            mention_keys_current_topic = list(mention_keys_by_topic_in_type[topic])
            positive_limit =  int(6*len(mention_keys_current_topic)**(1/2))
            negative_limit = positive_limit*15

            dict_positives_current_topic = dict_positive_pairs_by_topic[topic]
            dict_negatives_current_topic = dict_negative_pairs_by_topic[topic]

            if len(dict_positives_current_topic)>positive_limit:
                print(f"topic {topic} current positives {len(dict_positives_current_topic)} positive limit {positive_limit} ")
                sampled_keys = random.sample(dict_positives_current_topic.keys(), positive_limit)
                dict_positive_pairs_by_topic[topic] = {key: dict_positives_current_topic[key] for key in sampled_keys}

            if len(dict_negatives_current_topic)>negative_limit:
                print(f"topic {topic} current negatives {len(dict_negatives_current_topic)} negative limit {negative_limit} ")
                sampled_keys = random.sample(dict_negatives_current_topic.keys(), negative_limit)
                dict_negative_pairs_by_topic[topic] = {key: dict_negatives_current_topic[key] for key in sampled_keys}

    positive_pairs,negative_pairs = unnest_dictionary(dict_positive_pairs_by_topic), unnest_dictionary(dict_negative_pairs_by_topic)
    positive_labels, negative_labels= [1]*len(positive_pairs), [0]*len(negative_pairs)
    #Extract keys from pairs, not the similarity score
    all_pairs = list(positive_pairs.keys()) + list(negative_pairs.keys())
    all_labels = positive_labels + negative_labels
    print(f'there are {len(positive_pairs)} positive mention pairs and {len(negative_pairs)} negative mentions')
    return all_pairs, all_labels

def return_pairs_and_labels(cdcr_dataset, mention_type, additional_sampling):
    #TODO lift restriction on ECB topic 34 once the dataset is updated
    sentences_by_topic = cdcr_dataset.sentences_by_topic
    sentences_by_topic.pop('ECBplus-34',None)
    
    sentences_pairs_similarity_dict,dict_mentions_by_topic_in_type,dict_mention_pairs_by_pair = sentences_processing(sentences_by_topic,
                                                                               cdcr_dataset.mentions_by_sentence,
                                                                               cdcr_dataset.mentions_by_type[mention_type])
    # Previous method guarantees there is at least 1 mention of the chosen type in each sentence

    pairs,labels = create_mention_pairs(dict_sentences_pairs_similarity=sentences_pairs_similarity_dict,
                                                 mentions = cdcr_dataset.mentions, mention_pairs_by_pair=dict_mention_pairs_by_pair,
                                                 mention_keys_by_topic_in_type = dict_mentions_by_topic_in_type,
                                                 additional_sampling = additional_sampling)

    return pairs,labels