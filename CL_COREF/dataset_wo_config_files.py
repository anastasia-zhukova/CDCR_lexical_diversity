from collections import namedtuple, defaultdict
import json
import os
from typing import Dict, Any

class Mention:


    def __init__(self, mention_data: Dict[str, Any], type: str, source_path: str):
        self.type = type
        self.source_path = source_path
        self.new_context = mention_data.get('mention_context')

        # List of attributes to initialize
        attributes_to_initialize = [
            'coref_chain', 'mention_ner', 'mention_head_pos', 'mention_head_lemma',
            'mention_head', 'mention_head_id', 'doc_id', 'doc', 'is_continuous',
            'is_singleton', 'mention_id', 'mention_type', 'mention_full_type',
            'score', 'sent_id', 'old_context', 'tokens_number_context', 'tokens_number',
            'tokens_str', 'tokens_text', 'topic_id', 'topic', 'subtopic_id', 'subtopic',
            'coref_type', 'description', 'conll_doc_key'
        ]

        for attribute in attributes_to_initialize:
            setattr(self, attribute, mention_data.get(attribute))

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
        possible_sentences = [0,1,2]
        if self.sent_id in possible_sentences:
            possible_sentences.pop(self.sent_id)
        else:
            possible_sentences.pop()
        document_context = [value for _, value in sorted(doc_sentences_dict[possible_sentences[0]].items())] + [value for _, value in sorted(doc_sentences_dict[possible_sentences[1]].items())]
        self.new_context= (document_context,mention_context)



    def __repr__(self):
        return f"Mention(coref_chain={self.coref_chain}, tokens_str={self.tokens_str})"



class CDCR_Dataset:
    
    def __init__(self,name,data_path,context_method:str,special_mention_tokens:tuple=("[E]","[/E]")):

        self.mentions = {}
        self.name = name
        self.topics = []
        self.data_path=data_path
        self.sentences = {}
        self.mentions_by_topics = defaultdict(list)
        self.context_method = context_method
        self.special_mention_tokens=special_mention_tokens

        self.load_mentions()

    def load_mentions(self):
        entity_types = {"/entity_mentions.json":"entity",
                        "/event_mentions.json":"event"} 

        if not isinstance(self.data_path, list):
            self.data_path= [self.data_path]

        for path in self.data_path:

            # Load docs and topics
            docs, topics = self.read_conll_file(path)
            self.topics.extend(topics)
            #Load sentences
            self.sentences=self.make_dict_of_sentences(docs)

            for file_path, type_str in entity_types.items():
                mention_type_path = path + file_path
                if os.path.exists(mention_type_path):

                    ## Load mentions:
                    with open(mention_type_path, 'r') as f:
                        raw_mentions = json.load(f)
                    
                    for mention_dict in raw_mentions:
                        new_mention = Mention(mention_dict,type=type_str,source_path=path)

                        ## Validations
                        
                        
                        #conll_key exists in the CoNLL file?
                        if new_mention.conll_doc_key not in self.sentences.keys():
                            raise ValueError(f"Mention {new_mention.mention_id} does not have a matching CoNLL doc key with the dictionary of sentences")
                        

                        #mention already added to the dictionary?
                        if new_mention.mention_id in self.mentions:
                            raise ValueError(f"Mention '{new_mention.mention_id }' already exists in the dictionary of mentions.")
                        
                        #new_mention.tokens_text equals the concatenation of tokens extracted from the sentence?
                        new_mention_sentence = self.sentences[new_mention.conll_doc_key][new_mention.sent_id]

                        tokens_from_sentence = []
                        for token_mention_id in new_mention.tokens_number:
                            tokens_from_sentence.append(new_mention_sentence[token_mention_id])

                        differences = new_mention.validate_token_str(tokens_from_sentence)

                        if differences:
                            raise ValueError(f"Mention {new_mention.mention_id} has attribute tokens_text={new_mention.tokens_text} and the tokens extracted from the sentence are {tokens_from_sentence} in position {differences}")
                        

                        ## Assignments

                        #Create context
                        new_mention.create_context(doc_sentences_dict=self.sentences[new_mention.conll_doc_key],method=self.context_method,special_mention_tokens=self.special_mention_tokens)
                        #Store in dictionaries
                        self.mentions[new_mention.mention_id]=new_mention
                        self.mentions_by_topics[new_mention.topic_id].append(new_mention.mention_id)
                        

    def read_conll_file(self,doc_conll_path:str):
        Token = namedtuple("Token", "conll_doc_key sentence_id token_id token_str coref_chain")

        current_document_id = ''
        current_document_list = []
        dict_documents = {}
        doc_conll_path = os.path.join(doc_conll_path, "dataset.conll")
        with open(doc_conll_path, 'r', encoding='utf-8') as file:
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
    
    def make_dict_of_sentences(self, documents: dict) -> Dict:
        doc_dict = {}
        for doc, tokens in documents.items():
            token_dict = defaultdict(dict)
            for token in tokens:
                token_dict[token.sentence_id][token.token_id] = token.token_str
            doc_dict[doc] = token_dict

        return doc_dict