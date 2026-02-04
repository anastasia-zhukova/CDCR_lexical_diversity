from collections import namedtuple, defaultdict
import csv
import json
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
import string
import sys
import time
from torch.utils.data import Dataset, Subset
import traceback
from typing import Dict, Any
from sentence_transformers import SentenceTransformer

from cluster_documents import return_mentions_with_predicted_topics
nltk.download('stopwords')

class Mention:


    def __init__(self, mention_data: Dict[str, Any], m_type: str, source_path: str):
        self.type = m_type
        self.source_path = source_path
        self.new_context = mention_data.get('mention_context')

        # List of attributes to initialize
        attributes_to_initialize = [
            'coref_chain', 'mention_ner', 'mention_head_pos', 'mention_head_lemma',
            'mention_head', 'mention_head_id', 'doc_id', 'doc', 'is_continuous',
            'is_singleton', 'mention_id', 'mention_type', 'mention_full_type',
            'score', 'sent_id', 'mention_context', 'tokens_number_context', 'tokens_number',
            'tokens_str', 'tokens_text', 'topic_id', 'topic', 'subtopic_id', 'subtopic',
            'coref_type', 'description', 'conll_doc_key'
        ]

        for attribute in attributes_to_initialize:
            setattr(self, attribute, mention_data.get(attribute))
            
        self.conll_sent_key = f'{self.conll_doc_key}/{self.sent_id}'
        mention_wo_stopwords = [token for token in self.tokens_text if token == self.mention_head or (token not in string.punctuation and token not in stopwords.words("english"))]        
        if not len(mention_wo_stopwords):
            print(f"mention {self.mention_id} ignored for phrasing diversity. Head '{self.mention_head}' not in list of tokens {self.tokens_text}")
            self.mention_wo_stopwords = None
            self.mention_wordset = None

        else:
            mention_wordset = frozenset(mention_wo_stopwords)
            self.mention_wo_stopwords = mention_wo_stopwords
            self.mention_wordset = mention_wordset

        


    def validate_token_str(self, tokens_from_sentence:list) ->list:
        differences = []

        # Check each element at the same index in both lists, ignoring leading and trailing whitespace
        for i, (elem1, elem2) in enumerate(zip(self.tokens_text, tokens_from_sentence)):
            # Strip leading and trailing whitespace before comparison
            if elem1.strip() != elem2.strip():
                differences.append(i)

        return differences

    def create_context(self, doc_sentences_dict:dict,method:str,special_mention_tokens:tuple):
        
        mention_context = [value for _, value in sorted(doc_sentences_dict[self.sent_id].items())]

        # Determine the lower and upper limits for special mention tokens
        lower_limit, upper_limit = min(self.tokens_number), max(self.tokens_number)

        mention_context.insert(upper_limit+1,special_mention_tokens[1])
        mention_context.insert(lower_limit,special_mention_tokens[0])

        #Document context
        possible_sentences = [0,1,2,3,4,5,6,7,8] #Increased to 4 to minimize padding and replace it with tokens from the next sentence
        if self.sent_id in possible_sentences:
            possible_sentences.pop(self.sent_id)
        else:
            possible_sentences.pop()

        document_context1 = [value for _, value in sorted(doc_sentences_dict[possible_sentences[0]].items())] + [value for _, value in sorted(doc_sentences_dict[possible_sentences[1]].items())]
        document_context2 = [value for _, value in sorted(doc_sentences_dict[possible_sentences[2]].items())] + [value for _, value in sorted(doc_sentences_dict[possible_sentences[3]].items())]
        document_context3 = [value for _, value in sorted(doc_sentences_dict[possible_sentences[4]].items())] + [value for _, value in sorted(doc_sentences_dict[possible_sentences[5]].items())]
        document_context4 = [value for _, value in sorted(doc_sentences_dict[possible_sentences[6]].items())] + [value for _, value in sorted(doc_sentences_dict[possible_sentences[7]].items())]
        full_context = document_context1 + document_context2 + document_context3 + document_context4
        self.new_context= (full_context,mention_context)
        
    def return_gold_topic_representation(self)->dict:
        return {
            "mention_id": self.mention_id,
            "subtopic_id": self.subtopic_id,
            "coref_chain": self.coref_chain,
            "mention_type": self.type,
            "mention_head":self.mention_head,
            "mention_wo_stopwords": self.mention_wo_stopwords,
            "mention_wordset":self.mention_wordset
            
        }

    def __repr__(self):
        return f"Mention(coref_chain={self.coref_chain}, tokens_str={self.tokens_str})"



class CDCR_Dataset:
    
    def __init__(self,name,data_path,context_method:str,special_mention_tokens:tuple=("[E]","[/E]"),predicted_topics=False):

        self.mentions = {}
        self.docs={}
        self.name = name
        self.data_path=data_path
        self.sentences = {}
        self.mentions_by_topic = defaultdict(list)
        self.mentions_by_sentence = defaultdict(set)
        self.mentions_by_type = defaultdict(set)
        self.mentions_gold_topic_repr = []
        self.sentences_by_topic = defaultdict(dict)
        self.context_method = context_method
        self.special_mention_tokens=special_mention_tokens

        self.load_mentions()

        #Create gold clusters dataframe
        gold_clusters_df = pd.DataFrame(self.mentions_gold_topic_repr)
        gold_clusters_df["gold_cluster"] = pd.Categorical(gold_clusters_df["coref_chain"]).codes
        self.gold_clusters_df = gold_clusters_df

        #Change gold subtopics to predicted topics for ECB+
        if predicted_topics and name=="ecbplus_test":
            print(f"changing gold subtopics to predicted subtopics in {name}")
            self.mentions = return_mentions_with_predicted_topics(self.docs,self.mentions,compare_with_true_labels=True)

    def load_mentions(self):
        entity_types = {"/entity_mentions.json":"entity",
                        "/event_mentions.json":"event"}

        if not isinstance(self.data_path, list):
            self.data_path= [self.data_path]

        for path in self.data_path:
            # Get dataset name from path (e.g., "ECBplus" from "CL_COREF/data/ECBplus/train")
            # The parent directory name is the dataset name
            parent_dir = os.path.dirname(path)
            dataset_name = os.path.basename(parent_dir)

            # Load documents from parent directory's parquet file
            docs = self.read_parquet_documents(parent_dir)
            self.docs.update(docs)

            # Process dictionary of sentences by document id
            new_sentences_dict = self.make_dict_of_sentences(docs)

            # Create dictionary of sentences by topic to compute similarity within groups
            for doc_key, doc in new_sentences_dict.items():
                topic_num = doc_key.split("/")[0]
                dataset_topic_key = f"{dataset_name}-{topic_num}"

                for index, sentence in doc.items():
                    # key equals doc_key + sentence key and values are concatenated tokens
                    self.sentences_by_topic[dataset_topic_key].update({f"{doc_key}/{index}":' '.join(sentence.values())})

            # Add to the existing dictionary
            current_sentences_dict = self.sentences.copy()
            current_sentences_dict.update(new_sentences_dict)
            self.sentences = current_sentences_dict

            # Load mentions from JSON files in the train/val/test subdirectory
            for file_path, type_str in entity_types.items():
                mention_type_path = path + file_path
                if os.path.exists(mention_type_path):

                    ## Process mentions:
                    with open(mention_type_path, 'r') as f:
                        raw_mentions = json.load(f)

                    for mention_dict in raw_mentions:
                        new_mention = Mention(mention_dict, m_type=type_str, source_path=path)

                        ## Validations before storing the mention

                        # conll_key exists in the document dictionary?
                        if new_mention.conll_doc_key not in self.sentences.keys():
                            raise ValueError(f"Mention {new_mention.mention_id} does not have a matching doc key in the dictionary of sentences")

                        # mention already added to the dictionary?
                        if new_mention.mention_id in self.mentions:
                            raise ValueError(f"Mention '{new_mention.mention_id}' already exists in the dictionary of mentions.")

                        # new_mention.tokens_text equals the concatenation of tokens extracted from documents?
                        new_mention_sentence = self.sentences[new_mention.conll_doc_key][new_mention.sent_id]
                        tokens_from_sentence = [new_mention_sentence[token_mention_id] for token_mention_id in new_mention.tokens_number]
                        differences = new_mention.validate_token_str(tokens_from_sentence)

                        if differences:
                            raise ValueError(f"Mention {new_mention.mention_id} has attribute tokens_text={new_mention.tokens_text} and the tokens extracted from the sentence are {tokens_from_sentence} in position {differences}")

                        ## Assignments

                        # Create context
                        new_mention.create_context(doc_sentences_dict=self.sentences[new_mention.conll_doc_key], method=self.context_method, special_mention_tokens=self.special_mention_tokens)

                        # Grouping by other variables
                        self.mentions[new_mention.mention_id] = new_mention
                        self.mentions_by_sentence[new_mention.conll_sent_key].add(new_mention.mention_id)
                        self.mentions_by_type[new_mention.type].add(new_mention.mention_id)
                        self.mentions_gold_topic_repr.append(new_mention.return_gold_topic_representation())

                        # Grouping by dataset name and topic id
                        dataset_topic_key = f"{dataset_name}-{new_mention.topic_id}"
                        if dataset_topic_key not in self.sentences_by_topic.keys():
                            raise ValueError(f"There is a mismatch between the dataset_topic_key {dataset_topic_key} from mention {new_mention.mention_id} and these keys {self.sentences_by_topic.keys()}")

                        self.mentions_by_topic[dataset_topic_key].append(new_mention.mention_id)

        


    def read_parquet_documents(self, data_path: str) -> dict:
        """Read documents from parquet file and convert to expected format"""
        Token = namedtuple("Token", "conll_doc_key sentence_id token_id token_str coref_chain")

        dict_documents = {}
        parquet_path = os.path.join(data_path, "all_documents.parquet")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Read parquet file
        df_docs = pd.read_parquet(parquet_path)

        # Group by document
        for doc_key, doc_group in df_docs.groupby('topic/subtopic_name/doc'):
            token_list = []
            for _, row in doc_group.iterrows():
                token = Token(
                    conll_doc_key=doc_key,
                    sentence_id=row['sent_id'],
                    token_id=row['token_id'],
                    token_str=row['token'],
                    coref_chain='-'  # Not used in current implementation
                )
                token_list.append(token)

            dict_documents[doc_key] = token_list

        return dict_documents
    
    def make_dict_of_sentences(self, documents: dict) -> Dict:
        doc_dict = {}
        for doc, tokens in documents.items():
            token_dict = defaultdict(dict)
            for token in tokens:
                token_dict[token.sentence_id][token.token_id] = token.token_str
            doc_dict[doc] = token_dict

        return doc_dict



class CL_Dataset(Dataset):

    def __init__(self, name, data_paths, context_method, mention_type, tokenizer,
                 special_mention_tokens, mode="train",
                 additional_sampling=False, predicted_topics=False, sentence_transformer_model=None):
        #I will tokenize all the sequences up-front
        from sampling import return_pairs_and_labels
        self.cdcr_dataset = CDCR_Dataset(name=name, data_path=data_paths, context_method=context_method,
                                         special_mention_tokens=special_mention_tokens, predicted_topics=predicted_topics)

        #Tokenize mentions
        self.mention_type = mention_type
        self.tokenizer = tokenizer
        self.tokenized_mentions = self.tokenize_mentions()
        self.mode = mode

        #Load pair keys and labels
        if mode == "train":
            # Load sentence transformer model if provided
            if sentence_transformer_model is None:
                raise ValueError("sentence_transformer_model must be provided for train mode")
            st_model = SentenceTransformer(sentence_transformer_model)

            start_time = time.time()
            self.all_pairs, self.all_labels = return_pairs_and_labels(
                cdcr_dataset=self.cdcr_dataset,
                additional_sampling=additional_sampling,
                model=st_model
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time elapsed for sampling: {elapsed_time} seconds")
            
        elif mode == 'inference':
            self.keys_list = list(self.tokenized_mentions.keys())
        else:
            raise ValueError("Invalid mode. Supported values are 'train' and 'inference'.")

        #Here in this processing I should tokenize first, remove tokens in excess keeping the limit to 128, then concatenate 
    def tokenize_mentions(self):
        #Validate if special token [E] is recognized by the tokenizer
        start_token = self.cdcr_dataset.special_mention_tokens[0]
        end_token = self.cdcr_dataset.special_mention_tokens[1]

        mention_start = self.tokenizer.encode(start_token, add_special_tokens=False)
        mention_end = self.tokenizer.encode(end_token, add_special_tokens=False)

        if len(mention_start) != 1 or len(mention_end) != 1:
            raise Exception("Either the special token [E] or [/E] is splitted in more than 1 token by the subword tokenizer")

        # Create a dictionary to store the tokens
        tokenized_dict = {}
        #Tokenize mention representation with context
        mention_keys = self.cdcr_dataset.mentions_by_type[self.mention_type].intersection(self.cdcr_dataset.mentions.keys())
        if not mention_keys:
            print(f"dataset {self.cdcr_dataset.name} has no mentions of type {self.mention_type}")
            return tokenized_dict
        mention_keys = list(mention_keys)
        context_texts = [" ".join(self.cdcr_dataset.mentions[key].new_context[0]) for key in mention_keys]
        mention_texts = [" ".join(self.cdcr_dataset.mentions[key].new_context[1]) for key in mention_keys]
        #Remove linebreaks
        context_texts = [s.replace('\\n', '') for s in context_texts]
        mention_texts = [s.replace('\\n', '') for s in mention_texts]
        try:
            encoded_text = self.tokenizer(text=context_texts,text_pair=mention_texts,padding="max_length",max_length=120,truncation="only_first")
        except Exception as e:
            print(f"Tokenization of {self.cdcr_dataset.name} failed with the following error: {e}")
            # Print the full traceback
            traceback.print_exc()

            with open(f"context_texts_{self.cdcr_dataset.name}", mode='w', newline='') as file:
                # Create a CSV writer object
                csv_writer = csv.writer(file)
                # Write the list of strings to the CSV file
                csv_writer.writerow(context_texts)
            with open(f"mention_texts_{self.cdcr_dataset.name}", mode='w', newline='') as file:
                # Create a CSV writer object
                csv_writer = csv.writer(file)
                # Write the list of strings to the CSV file
                csv_writer.writerow(mention_texts)
            
            print("csv debug files writen in main folder")
            sys.exit("Exiting due to tokenization error.")

        for key in mention_keys: #Iterate over the encoded text based on the position of the key in the list
            # Access corresponding values in encoded_text
            input_ids = encoded_text["input_ids"][mention_keys.index(key)]
            attention_mask = encoded_text["attention_mask"][mention_keys.index(key)]
            
            input_ids_length = len(input_ids)
            if input_ids_length>120: #When first_only is activated and the first sentence is longer than max_value, the tokenizer returns the encoding without truncation
                
                i = -1
                #Remove tokens from mention until pair encoding reaches max_len
                while input_ids_length>120:
                    temp_mention = " ".join(self.cdcr_dataset.mentions[key].new_context[1][:i]).replace('\\n', '')
                    temp_context = " ".join(self.cdcr_dataset.mentions[key].new_context[0]).replace('\\n', '')
                    temp_encoding = self.tokenizer(text=temp_context,text_pair=temp_mention,padding="max_length",max_length=120,truncation="only_first")

                    input_ids = temp_encoding["input_ids"]
                    attention_mask = temp_encoding["attention_mask"]
                    input_ids_length = len(input_ids)
                    i = i-1

            #Validate that the special token [E] and [/E] were not removed due to truncation
            if input_ids.count(mention_start[0])!=1 or input_ids.count(mention_end[0])!=1:
                raise Exception(f"The truncation process removed either the [E] or the [/E] token. [E] occurs {input_ids.count(mention_start[0])} and [/E] occurs {input_ids.count(mention_end[0])} times in mention {key}")
                
            # Create a dictionary entry with the key and corresponding tensors
            tokenized_dict[key] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
                #Roberta Tokenizer does not have token type ids
            }
        
        return tokenized_dict

          
    def decode_input_ids(self,input_id):
        return self.tokenizer.decode(input_id)


    def __len__(self):
        if self.mode=="train":
            return len(self.all_labels)
        else: 
            return len(self.tokenized_mentions)
        
    def __getitem__(self, index): 
        if self.mode=="train":
            key_mention1, key_mention2= self.all_pairs[index]
            return self.tokenized_mentions[key_mention1],self.tokenized_mentions[key_mention2], self.all_labels[index]
        
        else: #Inference
            key = self.keys_list[index]
            return index, self.tokenized_mentions[key]
        

class CustomSubset(Subset):
    def __init__(self, dataset, indices,mode):
        super(CustomSubset, self).__init__(dataset, indices)
        if mode=="inference":
            self.keys_list = [self.dataset.keys_list[i] for i in self.indices]
        if mode =="train":
            self.all_pairs = [self.dataset.all_pairs[i] for i in self.indices]
        self.cdcr_dataset = self.dataset.cdcr_dataset