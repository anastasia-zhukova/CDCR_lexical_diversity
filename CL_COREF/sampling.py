from collections import defaultdict
from itertools import product
import numpy as np
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

def create_pairs(dict_sentences_pairs_similarity,mentions,mentions_by_sentence, mention_type):
    dict_positive_pairs_by_topic = defaultdict(dict)
    dict_negative_pairs_by_topic = defaultdict(dict)

    for topic, sentence_pairs in dict_sentences_pairs_similarity.items():

        if not sentence_pairs: 
            print(f"topic {topic} does not have sentence pairs")
            continue
        
        topic_sentence_similarity_median = statistics.median(list(sentence_pairs.values()))

        for (sentence_key1, sentence_key2), score in sentence_pairs.items():
            mentions_sentence1 = mentions_by_sentence[sentence_key1]
            mentions_sentence2 = mentions_by_sentence[sentence_key2]

            #Checks the combination of all mentions for each pair of sentences
            for mention_key1,mention_key2 in product(mentions_sentence1, mentions_sentence2):

                mention1 = mentions[mention_key1]
                mention2 = mentions[mention_key2]

                if mention1.type!=mention_type or mention2.type!=mention_type: # No cross-type comparison
                    continue

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
        #Return unnested dictionaries, no longer by topic
    return unnest_dictionary(dict_positive_pairs_by_topic), unnest_dictionary(dict_negative_pairs_by_topic)


def return_pairs_and_labels (cdcr_dataset, mention_type):
    sentences_pairs_similarity_dict = return_sentence_pair_similarity_by_topic(cdcr_dataset.sentences_by_topic,cdcr_dataset.mentions_by_sentence) #Dict sentence by 
    
    positive_pairs,negative_pairs = create_pairs(dict_sentences_pairs_similarity=sentences_pairs_similarity_dict,
                                    mentions=cdcr_dataset.mentions,mentions_by_sentence=cdcr_dataset.mentions_by_sentence, mention_type=mention_type)

    positive_labels, negative_labels= [1]*len(positive_pairs), [0]*len(negative_pairs)

    all_pairs = positive_pairs + negative_pairs
    all_labels = positive_labels + negative_labels

    return all_pairs,all_labels