from collections import defaultdict
from itertools import product
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import statistics
import torch

def return_sentence_pair_similarity_by_topic(dict_sentences_by_topic,mentions_by_sentence):
    model = SentenceTransformer('all-mpnet-base-v2')

    dict_sentences_pair_similarity_by_topic = {}
    for topic,sentences_dict in dict_sentences_by_topic.items(): #Now this come from the dataset
        print(f'current topic {topic}')
        sentences = list(sentences_dict.values())
        list_of_keys = list(sentences_dict.keys())

        #Compute similarity
        if not sentences:
            print(f"topic {topic} is empty")
            continue

        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)

        # Get the indices of the lower triangular elements of the similarity matrix
        indices = torch.tril(torch.ones_like(cosine_scores), diagonal=-1).nonzero()

        # Create a dictionary using the indices
        lower_triangle_dict = {}

        for i, j in indices:
            key_sentence1 = list_of_keys[i.item()]
            key_sentence2 = list_of_keys[j.item()]
            key_pair = (key_sentence1, key_sentence2) 

            #Ignore sentences without mentions
            if not mentions_by_sentence[key_sentence1] or not mentions_by_sentence[key_sentence2]:
                continue

            similarity_score = round(cosine_scores[i, j].item(), 3)
            if similarity_score<1: #ignores those sentences with similarity=1. Those are mostly exact citations
                lower_triangle_dict[key_pair] = similarity_score

        dict_sentences_pair_similarity_by_topic[topic] = lower_triangle_dict

    return dict_sentences_pair_similarity_by_topic

def unnest_dictionary(input_dict,key=''):
  
    unnested_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            unnested_dict.update(unnest_dictionary(value, key))
        else:
            unnested_dict[key] = value

    return unnested_dict

def create_pairs(dict_sentences_pairs_similarity,cdcr_dataset, mention_type, additional_sampling = False)->dict:

    mentions=cdcr_dataset.mentions
    mentions_by_sentence=cdcr_dataset.mentions_by_sentence
    mentions_by_type = cdcr_dataset.mentions_by_type[mention_type]
    #Type and topic require filtering

    dict_positive_pairs_by_topic = defaultdict(dict)
    dict_negative_pairs_by_topic = defaultdict(dict)

    for topic, sentence_pairs in dict_sentences_pairs_similarity.items():

        if not sentence_pairs: 
            print(f"topic {topic} does not have sentence pairs")
            continue
        
        topic_sentence_similarity_median = statistics.median(list(sentence_pairs.values()))
        mention_keys_current_topic = []
        for (sentence_key1, sentence_key2), score in sentence_pairs.items():
            mentions_sentence1 = mentions_by_sentence[sentence_key1]
            mentions_sentence2 = mentions_by_sentence[sentence_key2]

            #Checks the combination of all mentions for each pair of sentences
            for mention_key1,mention_key2 in product(mentions_sentence1, mentions_sentence2):

                mention1 = mentions[mention_key1]
                mention2 = mentions[mention_key2]

                if mention1.type!=mention_type or mention2.type!=mention_type: # No cross-type comparison
                    continue
                mention_keys_current_topic.append(mention_key1)
                same_coref_chain = mention1.coref_chain == mention2.coref_chain
                new_dict_entry = {(mention_key1,mention_key2):score}

                if same_coref_chain:
                    dict_positive_pairs_by_topic[topic].update(new_dict_entry)
                else:
                    dict_negative_pairs_by_topic[topic].update(new_dict_entry)

            
        ## Subsampling
        positive_pairs_scores= list(dict_positive_pairs_by_topic[topic].values())

        #Downsample negative pairs based on positive pairs score's Q3
        third_quartile = np.percentile(positive_pairs_scores, 75)
        dict_negative_pairs_by_topic[topic] = {key: value for key, value in dict_negative_pairs_by_topic[topic].items() if value > third_quartile}
        
        #Downsample positive pairs based on topic's median
        dict_positive_pairs_by_topic[topic] = {key: value for key, value in dict_positive_pairs_by_topic[topic].items() if value < topic_sentence_similarity_median}

        #Second subsampling based on mention's number (6sqrt(n))
        if additional_sampling:
            mention_keys_current_topic = list(set(mention_keys_current_topic))
            positive_limit =  int(6*len(mention_keys_current_topic)**(1/2))
            negative_limit = positive_limit*20

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


        #Return unnested dictionaries, no longer by topic
    return unnest_dictionary(dict_positive_pairs_by_topic), unnest_dictionary(dict_negative_pairs_by_topic)

def return_pairs_and_labels(cdcr_dataset, mention_type, additional_sampling):
    #TODO lift restriction on ECB topic 34 once the dataset is updated
    sentences_by_topic = cdcr_dataset.sentences_by_topic
    sentences_by_topic.pop('ECBplus-34',None)
    sentences_pairs_similarity_dict = return_sentence_pair_similarity_by_topic(sentences_by_topic,cdcr_dataset.mentions_by_sentence) #Dict sentence by 
    # Remove ECB topic 34
    positive_pairs,negative_pairs = create_pairs(dict_sentences_pairs_similarity=sentences_pairs_similarity_dict,
                                                 cdcr_dataset = cdcr_dataset, mention_type=mention_type,
                                                 additional_sampling = additional_sampling)

    positive_labels, negative_labels= [1]*len(positive_pairs), [0]*len(negative_pairs)
    #Extract keys from pairs, not the similarity score
    all_pairs = list(positive_pairs.keys()) + list(negative_pairs.keys())
    all_labels = positive_labels + negative_labels
    all_scores = list(positive_pairs.values()) + list(negative_pairs.values())

    return all_pairs,all_labels,all_scores