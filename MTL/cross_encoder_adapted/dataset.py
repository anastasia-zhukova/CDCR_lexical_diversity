from datasets import load_dataset, DatasetDict
import torch
from torch.utils import data
import collections
from itertools import combinations
import numpy as np
import os
import json

import hashlib
from collections import namedtuple

from cluster_documents import return_mentions_with_predicted_topics

class DatasetConfig():
    def __init__(self, data_folder,mention_type,use_gold_mentions=True):
        self.data_folder = data_folder
        self.use_gold_mentions= use_gold_mentions
        self.mention_type = mention_type


def generate_unique_hash(text):
    hash_object = hashlib.sha256(text.encode())
    hex_digest = hash_object.hexdigest()
    decimal_hash = int(hex_digest, 16)  # Convert hexadecimal to decimal
    hash_value = decimal_hash / (10 ** len(hex_digest))  # Normalize the hash value to a float between 0 and 1
    return hash_value

def read_conll_file(filename):
    Token = namedtuple("Token", "conll_doc_key sentence_id token_id token_str coref_chain")

    current_document_id = ''
    current_document_list = []
    dict_documents = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("#begin") or line == '':
                continue  # Skip the header line

            elif line.startswith("#end") or line == '':
                current_document_id = current_document_list[0].conll_doc_key
                dict_documents[current_document_id] = current_document_list
                current_document_list = []
            else:
                parts = line.split('\t')
                token = Token(parts[0],int(parts[1]),int(parts[2]),parts[3],
                              parts[4].replace('(','').replace(')','')  )

                current_document_list.append(token)

    topics = []
    for key in dict_documents.keys():
        topics.append(key.split('/')[0])
    topics = list(set(topics))
    
    return dict_documents, topics

class CrossEncoderDataset(data.Dataset):
    def __init__(self, mentions_repr, first, second, labels):
        self.instances = [' '.join([mentions_repr[first[i]], "[SEP]",
                                    mentions_repr[second[i]], "[SEP]"]) for i in range(len(first))]
        self.labels = labels.to(torch.float)


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)




class CrossEncoderDatasetInstances(data.Dataset):
    def __init__(self, instances):
        self.instances = instances


    def __len__(self):
        return len(self.instances)


    def __getitem__(self, index):
        return self.instances[index]



class CrossEncoderDatasetFull(data.Dataset):

    def __init__(self, config,split_name='train', same_lemma=False):
        self.dataset_id = "/".join([config.data_folder.split('/')[-3],config.mention_type,split_name])
        self.read_files(config,split_name) #Sets self.documents, self.topics and self.mentions belonging to the topics
        # self.lemmas = np.asarray([x['lemmas'] for x in self.mentions]) #Do not know yet why we need the lemmas}

        #If the mention labels are only turned into tensors to create the labels then I do not
        #need to use the generate_hash function later
        self.coreference_labels = torch.tensor([generate_unique_hash(m['coref_chain']) for m in self.mentions])
        self.doc_dict = self.make_dict_of_sentences(self.documents)

        #Initialize mentions by topics list and pairs lists
        self.mentions_by_topics = collections.defaultdict(list)
        self.first = []
        self.second = []
        self.labels = []

        #Don't know why they do this
        # if same_lemma:
        #     idx = (self.lemmas[self.first] == self.lemmas[self.second]).nonzero()
        #     self.first = self.first[idx]
        #     self.second = self.second[idx]
        #     self.labels = self.labels[idx]
        
        self.instances = []
        self.instance_ids = []
        self.task='CDCR'

    def assign_mentions_with_new_topics(self, modified_mentions):
        self.mentions = modified_mentions

    def create_mention_pairs_by_topics(self,use_predicted=False):

        for i, m in enumerate(self.mentions):
            if use_predicted:
                self.mentions_by_topics[m['predicted_topic']].append(i)
            else:
                self.mentions_by_topics[m['topic']].append(i)

        for _, mentions in self.mentions_by_topics.items():
            first, second = zip(*list(combinations(range(len(mentions)), 2)))
            mentions = torch.tensor(mentions)
            first, second = torch.tensor(first), torch.tensor(second)
            first, second = mentions[first], mentions[second]
            labels = (self.coreference_labels[first] != 0) & (self.coreference_labels[second] != 0) \
                     & (self.coreference_labels[first] == self.coreference_labels[second])
            labels = torch.tensor(labels,dtype=torch.float)

            self.first.append(first)
            self.second.append(second)
            self.labels.append(labels)

        self.first = torch.cat(self.first)
        self.second = torch.cat(self.second)
        self.labels = torch.cat(self.labels)


    def negative_sampling(self):
        # Downsample the number of negative examples following a 1:5 proportion 
        # https://github.com/ariecattan/coref/blob/master/train_pairwise_scorer.py
        positives = torch.nonzero(self.labels == 1).squeeze()
        positive_ratio = len(positives) / len(self.first)
        negatives = torch.nonzero(self.labels != 1).squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 5).to(torch.long)
        sampled_negatives = negatives[torch.nonzero(rands).squeeze()]

        new_first = torch.cat((self.first[positives], self.first[sampled_negatives]))
        new_second = torch.cat((self.second[positives], self.second[sampled_negatives]))
        new_labels = torch.cat((self.labels[positives], self.labels[sampled_negatives]))

        self.first, self.second, self.labels = new_first, new_second, new_labels


        
    def read_files(self, config,split_name):
        docs_path = os.path.join(config.data_folder + split_name +'/','dataset.conll')
        mentions_path = os.path.join(config.data_folder+ split_name +'/','{}.json'.format(config.mention_type))

        self.documents, self.topics  = read_conll_file(docs_path)

        self.mentions = []
        if config.use_gold_mentions:
            with open(mentions_path, 'r') as f:
                self.mentions = json.load(f)
        #Filter mentions that belong to the topics of the current split
        self.mentions = [mention for mention in self.mentions if mention['topic_id'] in self.topics]
        #Filter language of mentions to english for those mentions with that property
        if "language" in self.mentions[0].keys():
            self.mentions = [mention for mention in self.mentions if mention['language']=="english"]

    #Modified method
    def make_dict_of_sentences(self, documents):
        doc_dict = {}
        for doc, tokens in documents.items():
            dict = collections.defaultdict(list)
            for token in tokens:

                dict[token.sentence_id].append([token.token_id, token.sentence_id, 
                                                token.token_str, token.coref_chain])
            doc_dict[doc] = dict

        return doc_dict


    def encode_mention_with_context(self, mention):
        doc_id, sentence_id = mention['conll_doc_key'], int(mention['sent_id']) #replaced doc_id with conll_doc_key

        try:
            tokens = self.doc_dict[doc_id][sentence_id]
        except KeyError:
            raise Exception(f"KeyError occurred for mention {mention['mention_id']} doc_id: {doc_id}, sentence_id: {sentence_id}")

        #I suppose these tokens are the tokens of the whole sentence
        token_ids = [x[0] for x in tokens]

        start_idx = token_ids.index(min(mention['tokens_number'])) #tokens_ids replaced by tokens_numbers
        end_idx = token_ids.index(max(mention['tokens_number'])) + 1

        # x[2] is the token string
        mention_repr = [x[2] for x in tokens[:start_idx]] + ["[START]"] \
                       + [x[2] for x in tokens[start_idx:end_idx]] + ["[END]"] \
                       + [x[2] for x in tokens[end_idx:]]
        #Then the context would be: 'Tara', 'Reid', 'has', 'entered', 'a', 'rehab', 'program', '.'
        #In that example, if the mention is entered, then it takes all the tokens from the sentence
        # prior the mention and those after:
        #  'went', 'to',  'Promises',  '[START]',  'on',  'Tuesday',  '[END]',  'and',  'that',  'her',  'friends',
        return ' '.join(mention_repr)


    def prepare_mention_representation(self, mentions):
        return np.asarray([self.encode_mention_with_context(m) for m in mentions])


    def prepare_pair_of_mentions(self):
        mentions_repr = np.asarray([self.encode_mention_with_context(m) for m in self.mentions])
        instances = [' '.join([mentions_repr[self.first[i]], "[SEP]",
                                    mentions_repr[self.second[i]], "[SEP]"]) for i in range(len(self.first))]
        # Then it turns it into:
        # 'A friend of the actress told People she went to Promises [START] on Tuesday [END] and that her friends and family supported her decision .'
        self.instances= instances

        #Create key with mentions ids
        self.instance_ids = ['[SEP]'.join([self.dataset_id, self.mentions[self.first[index]]['mention_id'],
                                   self.mentions[self.second[index]]['mention_id']])
                     for index in range(len(self.first))]


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1), self.instance_ids[index], self.task


## Paraphrase dataset
class ParaphraseDataset(data.Dataset):

    def __init__(self, config,split_name='train'):

        if config.data_folder=="mrpc":
            dataset_dict = load_dataset('glue',config.data_folder)
            dataset_name = "mrpc"
        
        else:
            dataset_dict = DatasetDict.load_from_disk(config.data_folder)
            dataset_name = "how"

        self.dataset = dataset_dict[split_name]
        self.dataset_id = "/".join([dataset_name,split_name])
        
        self.instances  = [f"{s1}[SEP]{s2}[SEP]" for s1, s2 in zip(self.dataset['sentence1'], self.dataset['sentence2'])]
        self.labels = torch.Tensor(self.dataset["label"])
        
        #Create key with index 
        self.instance_ids = [f"{self.dataset_id}[SEP]{s1}" for s1 in self.dataset['idx']]
        self.task='PI'



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1), self.instance_ids[index], self.task



### Mixed datasets class

class CrossEncoderMixedDatasetFull(data.Dataset):
    def __init__(self,config,datasets_config,experiment_split ):

        self.mentions = []

        self.instances = []
        self.labels = []
        self.instance_ids = []
        self.datasets_task = []


        for dataset_config_dict in experiment_split:
            dataset_key = dataset_config_dict['config_key']
            split = dataset_config_dict['split']
            get_topics =dataset_config_dict['get_topics']
            task = dataset_config_dict['task']

            temp_folder = datasets_config[dataset_key]['data_folder'] 
            temp_config = config.copy()
            temp_config.put('data_folder',temp_folder)

            if task=="CDCR":
                temp_mention_type = datasets_config[dataset_key]['mention_type']
                temp_config.put('mention_type',temp_mention_type)

            dataset_instance = None
            if task =="CDCR":
                dataset_instance = CrossEncoderDatasetFull(temp_config,split )
                self.mentions.extend(dataset_instance.mentions)
                if get_topics=="predict_topics":
                    mentions_w_predicted_topics = return_mentions_with_predicted_topics(dataset_instance)
                    dataset_instance.assign_mentions_with_new_topics(mentions_w_predicted_topics)
                    #predict topics

                dataset_instance.create_mention_pairs_by_topics(use_predicted=True)

                if get_topics=='negative_sampling':
                    dataset_instance.negative_sampling()
            
                dataset_instance.prepare_pair_of_mentions()


            if task=="PI":
                dataset_instance = ParaphraseDataset(temp_config,split )

            self.instances.extend(dataset_instance.instances)
            self.labels.extend(dataset_instance.labels)
            self.instance_ids.extend(dataset_instance.instance_ids)
            self.datasets_task.extend([dataset_instance.task]*len(dataset_instance.instances))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        instance = self.instances[index]
        label = self.labels[index]
        instance_id = self.instance_ids[index]
        task = self.datasets_task[index]

        return instance, label, instance_id, task







#Since I define the split in the initialization of the CrossEncoderDatasetFull class,
# I don't need this class, but I will leave it here in case that later
# I need to divide it by topic for evaluation
class CrossEncoderDatasetTopic(data.Dataset):
    def __init__(self, full_dataset, topic):
        super(CrossEncoderDatasetTopic, self).__init__()
        self.topic_mentions_ids = full_dataset.mentions_by_topics[topic]
        self.topic_mentions = [full_dataset.mentions[x] for x
                               in self.topic_mentions_ids]

        first, second = zip(*list(combinations(range(len(self.topic_mentions)), 2)))
        self.first, self.second = torch.tensor(first), torch.tensor(second)

        self.instances = full_dataset.prepare_pair_of_mentions(
            self.topic_mentions, self.first, self.second)



    def __len__(self):
        return len(self.topic_mentions)


    def __getitem__(self, index):
        return self.instances[index]
