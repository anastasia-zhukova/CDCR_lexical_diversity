#Classify docs into topics
import sklearn.cluster
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def return_mentions_with_predicted_topics(dataset,compare_with_true_labels=False):

    #Prepare data
    concatenated_documents_dict = {}
    for key, tokens in dataset.documents.items():
        concatenated_tokens = ' '.join(token.token_str for token in tokens)
        concatenated_documents_dict[key]= concatenated_tokens

    #Vectorize values
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1,3),
                                    stop_words='english')
    X = vectorizer.fit_transform(concatenated_documents_dict.values())
    #Create clusters
    NUM_TOPICS=20

    kmeans = sklearn.cluster.KMeans(n_clusters=NUM_TOPICS, init='k-means++', max_iter=200,
                                    n_init=20,random_state=665,algorithm='auto')

    kmeans.fit(X)

    if compare_with_true_labels:
        true_topic_labels = []
        for key in concatenated_documents_dict.keys():
            true_topic_labels.append(key.split('/')[0])

        labels_mapping = {}
        for label in dataset.topics:
            labels_mapping[label] = len(labels_mapping)

        true_labels_int = [labels_mapping[label] for label in true_topic_labels]

        #Print metrics
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels_int, kmeans.labels_))
        print("Completeness: %0.3f" % metrics.completeness_score(true_labels_int, kmeans.labels_))
        print("V-measure: %0.3f" % metrics.v_measure_score(true_labels_int, kmeans.labels_))
        print("Adjusted Rand-Index: %.3f"
                % metrics.adjusted_rand_score(true_labels_int, kmeans.labels_))
        
    
    #Modify mentions:
    if len(dataset.documents) != len(kmeans.labels_):
        raise ValueError("Dictionary of docs and list of labels must be of the same size")
    else:
        predicted_topics_dict = {key: value for key, value in zip(dataset.documents.keys(), kmeans.labels_)}

    modified_mentions = dataset.mentions.copy()
    for mention in modified_mentions:
        conll_doc_key = mention.get('conll_doc_key')
        if conll_doc_key is not None:
            predicted_topic = predicted_topics_dict.get(conll_doc_key)
            if predicted_topic is not None:
                mention['predicted_topic'] = predicted_topic
            else:
                raise ValueError("No predicted topic found")
        else:
            raise ValueError("No conll_doc_key found")
    
    return modified_mentions

def check_predicted_vs_true_clusters(true_labels_int, predicted_clusters):
        # Create a dictionary to store the correspondence between KMeans labels and unique true labels
    label_correspondence = {}
    true_label_to_kmeans_labels = {}  # To track true labels and their associated KMeans labels

    for true_label, kmeans_label in zip(true_labels_int, predicted_clusters):
        if kmeans_label not in label_correspondence:
            label_correspondence[kmeans_label] = set([true_label])
        else:
            label_correspondence[kmeans_label].add(true_label)

        if true_label not in true_label_to_kmeans_labels:
            true_label_to_kmeans_labels[true_label] = set([kmeans_label])
        else:
            true_label_to_kmeans_labels[true_label].add(kmeans_label)

    # Print the correspondence
    for kmeans_label, unique_true_labels in label_correspondence.items():
        print(f"KMeans Label {kmeans_label}: Unique True Labels {unique_true_labels}")

    print("\nTrue Labels with more than one KMeans Label:")
    for true_label, associated_kmeans_labels in true_label_to_kmeans_labels.items():
        if len(associated_kmeans_labels) > 1:
            print(f"True Label {true_label}: Associated KMeans Labels {associated_kmeans_labels}")

def elbow_method(vectorized_docs):
    import matplotlib.pyplot as plt

    # Determine a range of potential cluster numbers
    possible_clusters = range(2, 31)  # You can adjust the range as needed

    inertia_values = []

    # Calculate the inertia for each cluster number
    for num_clusters in possible_clusters:
        kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, init='k-means++', max_iter=200,
                                        n_init=20, random_state=665, algorithm='auto')
        kmeans.fit(vectorized_docs)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(possible_clusters, inertia_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Cluster Number')

    # Find the elbow point (where the rate of decrease slows down)
    diffs = [inertia_values[i] - inertia_values[i - 1] for i in range(1, len(inertia_values))]
    elbow_idx = diffs.index(max(diffs)) + 1
    plt.scatter(elbow_idx + 2, inertia_values[elbow_idx], color='red', marker='o', label='Elbow Point')

    plt.legend()
    plt.show()
    print(elbow_idx)