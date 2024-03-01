#Classify docs into topics
import sklearn.cluster
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def return_mentions_with_predicted_topics(docs,mentions,compare_with_true_labels=False):

    #Prepare data
    concatenated_documents_dict = {}
    for key, tokens in docs.items():
        concatenated_tokens = ' '.join(token.token_str for token in tokens)
        concatenated_documents_dict[key]= concatenated_tokens

    #Vectorize values
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1,3),
                                    stop_words='english')
    X = vectorizer.fit_transform(concatenated_documents_dict.values())

    #Define number of clusters

    NUM_TOPICS = 20
    kmeans = sklearn.cluster.KMeans(n_clusters=NUM_TOPICS, init='k-means++', max_iter=200,
                                    n_init=20,random_state=665,algorithm='auto')

    kmeans.fit(X)

    if compare_with_true_labels:
        true_topic_labels = []
        for key in concatenated_documents_dict.keys():
            true_topic_labels.append(key.split('/')[1])

        gold_subtopics = set([mention.subtopic_id for mention in mentions.values()])
        labels_mapping = {}
        for label in gold_subtopics:
            labels_mapping[label] = len(labels_mapping)

        true_labels_int = [labels_mapping[label] for label in true_topic_labels]

        #Print metrics
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels_int, kmeans.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(true_labels_int, kmeans.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(true_labels_int, kmeans.labels_))
        print("Adjusted Rand-Index: %.3f"
                % metrics.adjusted_rand_score(true_labels_int, kmeans.labels_))
        
    
    #Modify mentions:
    if len(docs) != len(kmeans.labels_):
        raise ValueError("Dictionary of docs and list of labels must be of the same size")
    else:
        predicted_topics_dict = {key: value for key, value in zip(docs.keys(), kmeans.labels_)}

    modified_mentions = {}
    for key,mention in mentions.items():
        conll_doc_key = mention.conll_doc_key
        if conll_doc_key is not None:
            predicted_topic = predicted_topics_dict.get(conll_doc_key)
            if predicted_topic is not None:
                mention.topic_id = str(predicted_topic)
                modified_mentions[key] = mention
            else:
                raise ValueError("No predicted topic found")
        else:
            raise ValueError("No conll_doc_key found")
    
    return modified_mentions

