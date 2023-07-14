import torch
from torch.utils import data
import collections
from itertools import combinations
import numpy as np
import os
import json

import hashlib
from collections import namedtuple

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
    with open(filename, 'r') as file:
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
    
    return dict_documents

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
    def __init__(self, config, same_lemma=False):
        # self.corpus = corpuss
        # self.documents = corpus.documents
        # self.mentions = corpus.mentions

        self.read_files(config)
        # self.lemmas = np.asarray([x['lemmas'] for x in self.mentions]) #Do not know yet why we need the lemmas}
        self.topics = set([m['topic'] for m in self.mentions])
        #If the mention labels are only turned into tensors to create the labels then I do not
        #need to use the generate_hash function later
        self.mention_labels = torch.tensor([generate_unique_hash(m['coref_chain']) for m in self.mentions])
        self.doc_dict = self.make_dict_of_sentences(self.documents)


        self.mentions_by_topics = collections.defaultdict(list)
        for i, m in enumerate(self.mentions):
            self.mentions_by_topics[m['topic']].append(i)

        self.first = []
        self.second = []
        self.labels = []

        for topic, mentions in self.mentions_by_topics.items():
            first, second = zip(*list(combinations(range(len(mentions)), 2)))
            mentions = torch.tensor(mentions)
            first, second = torch.tensor(first), torch.tensor(second)
            first, second = mentions[first], mentions[second]
            labels = (self.mention_labels[first] != 0) & (self.mention_labels[second] != 0) \
                     & (self.mention_labels[first] == self.mention_labels[second])

            self.first.extend(first)
            self.second.extend(second)
            self.labels.extend(labels)

        self.first = torch.tensor(self.first)
        self.second = torch.tensor(self.second)
        self.labels = torch.tensor(self.labels, dtype=torch.float)

        #Don't know why they do this
        # if same_lemma:
        #     idx = (self.lemmas[self.first] == self.lemmas[self.second]).nonzero()
        #     self.first = self.first[idx]
        #     self.second = self.second[idx]
        #     self.labels = self.labels[idx]



        self.instances = self.prepare_pair_of_mentions(self.mentions, self.first,
                                                       self.second)



    def read_files(self, config): # Removed split_name parameter since only ECB+ has split folders
        docs_path = os.path.join(config.data_folder,'dataset.conll')
        mentions_path = os.path.join(config.data_folder,'{}.json'.format(config.mention_type))
        # with open(docs_path, 'r') as f:
        #     self.documents = json.load(f)

        self.documents = read_conll_file(docs_path)
        self.mentions = []
        if config.use_gold_mentions:
            with open(mentions_path, 'r') as f:
                self.mentions = json.load(f)


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


    def prepare_pair_of_mentions(self, mentions, first, second):
        mentions_repr = np.asarray([self.encode_mention_with_context(m) for m in mentions])
        instances = [' '.join([mentions_repr[first[i]], "[SEP]",
                                    mentions_repr[second[i]], "[SEP]"]) for i in range(len(first))]
        # Then it turns it into:
        # 'A friend of the actress told People she went to Promises [START] on Tuesday [END] and that her friends and family supported her decision .'
        return instances



    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        return self.instances[index], self.labels[index].unsqueeze(-1)





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
