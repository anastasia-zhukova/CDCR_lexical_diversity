"""Some parts are borrowed from
https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
"""
from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return (0 if p + r == 0
            else (1 + beta * beta) * p * r / (beta * beta * p + r))


def evaluate_non_referrings(doc_non_referring_infos):
    tp, _tn, fp, fn = 0, 0, 0, 0

    for doc_id in doc_non_referring_infos:
        key_non_referrings, sys_non_referrings = doc_non_referring_infos[
                doc_id]
        for m in key_non_referrings:
            if m in sys_non_referrings:
                tp += 1
            else:
                fn += 1
        for m in sys_non_referrings:
            if m not in key_non_referrings:
                fp += 1

    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * recall * precision / (recall + precision)
            if (recall + precision) > 0 else 0)

    return recall, precision, f1


class Evaluator:
    def __init__(self, metric,dataset_name, beta=1, keep_aggregated_values=False,):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta
        self.keep_aggregated_values = keep_aggregated_values
        self.dataset_name=dataset_name

        if keep_aggregated_values:
            self.aggregated_p_num = []
            self.aggregated_p_den = []
            self.aggregated_r_num = []
            self.aggregated_r_den = []

    def update(self, coref_info,dict_phrasing_diversity):
        (key_clusters, sys_clusters, key_mention_sys_cluster,
                sys_mention_key_cluster) = coref_info

        if self.metric == ceafe or self.metric == ceafm:
            pn, pd, rn, rd = self.metric(sys_clusters, key_clusters)
        elif self.metric == lea:
            pn, pd = self.metric(sys_clusters, key_clusters,
                    sys_mention_key_cluster)
            rn, rd = self.metric(key_clusters, sys_clusters,
                    key_mention_sys_cluster)
        elif self.metric == pd_lea:
            pn, pd,predicted_clusters_pd_df = self.metric(sys_clusters, key_clusters,
                    sys_mention_key_cluster,dict_phrasing_diversity,"predicted")
            rn, rd,gold_clusters_pd_df = self.metric(key_clusters, sys_clusters,
                    key_mention_sys_cluster,dict_phrasing_diversity,"gold")

            combined_df = gold_clusters_pd_df.merge(predicted_clusters_pd_df, on='mention_id',how='left')
            combined_df['df'] = self.dataset_name
            combined_df[['df','mention_id','coref_chain','gold_label','pd_gold','predicted_label','pd_predicted']].to_csv(f'predictions/clusters_{self.dataset_name}.csv',index=False,sep=';',decimal=',')
        else:
            pn, pd = self.metric(sys_clusters, sys_mention_key_cluster)
            rn, rd = self.metric(key_clusters, key_mention_sys_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

        if self.keep_aggregated_values:
            self.aggregated_p_num.append(pn)
            self.aggregated_p_den.append(pd)
            self.aggregated_r_num.append(rn)
            self.aggregated_r_den.append(rd)

    def get_f1(self):
        return f1(self.p_num,
                self.p_den,
                self.r_num,
                self.r_den,
                beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

    def get_aggregated_values(self):
        return (self.aggregated_p_num, self.aggregated_p_den,
                self.aggregated_r_num, self.aggregated_r_den)


def evaluate_documents(doc_coref_infos,dict_phrasing_diversity, metric,dataset_name, beta=1):
    evaluator = Evaluator(metric,dataset_name, beta=beta)
    for doc_id in doc_coref_infos:
        evaluator.update(doc_coref_infos[doc_id],dict_phrasing_diversity)
    return (evaluator.get_recall(), evaluator.get_precision(),
            evaluator.get_f1())


def get_document_evaluations(doc_coref_infos, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta, keep_aggregated_values=True)
    for doc_id in doc_coref_infos:
        evaluator.update(doc_coref_infos[doc_id])
    return evaluator.get_aggregated_values()


def mentions(clusters, mention_to_gold):
    setofmentions = set(mention for cluster in clusters for mention in cluster)
    correct = setofmentions & set(mention_to_gold.keys())
    return len(correct), len(setofmentions)


def b_cubed(clusters, mention_to_gold):
    num, den = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[mention_to_gold[m]] += 1
        for c2 in gold_counts:
            correct += gold_counts[c2] * gold_counts[c2]

        num += correct / float(len(c))
        den += len(c)

    return num, den


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def phi3(c1, c2):
    return len([m for m in c1 if m in c2])

def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)

def ceafm(clusters, gold_clusters):
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi3(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(input_clusters, output_clusters, mention_to_gold):
    num, den = 0, 0

    for c in input_clusters:
        if len(c) == 1:
            all_links = 1
            if c[0] in mention_to_gold and len(
                    output_clusters[mention_to_gold[c[0]]]) == 1:
                common_links = 1
            else:
                common_links = 0
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            for i, m in enumerate(c):
                if m in mention_to_gold:
                    for m2 in c[i + 1:]:
                        if m2 in mention_to_gold and mention_to_gold[
                                m] == mention_to_gold[m2]:
                            common_links += 1
                        # else:
                        #    print('!! ', m2, '--', m2.get_span(), ' ',
                        #           m2.min_spans, ' ', mention_to_gold[m], ' ',
                        #           mention_to_gold[m2], ' ' ,
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m]]], ' -- ',
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m2]]])

        num += len(c) * common_links / float(all_links)
        den += len(c)

    return num, den



def phrasing_diversity_calc(mentions) -> float:
    """
    Calculates a metric that represents phrasing complexity of an entity, i.e., how various is the wording of the
    phrases referring to this entity. For more details see LREC paper
    Returns:
        Internal headword-based structure, value of the phrasing complexity


    """

    headwords_phrase_tree = {}
    for mention_head,mention_wordset,mention_wo_stopwords in mentions:
        if not mention_wo_stopwords:
            continue

        if mention_head not in headwords_phrase_tree:
            headwords_phrase_tree[mention_head] = {"set": {mention_wordset},
                                                            "list": [mention_wo_stopwords]}
        else:
            # set ensures unique phrases
            headwords_phrase_tree[mention_head]["set"].add(mention_wordset)
            # list keeps actual number of phrase occurence
            headwords_phrase_tree[mention_head]["list"].append(mention_wo_stopwords)

    sets = []
    fractions = []
    for _, head_properties in headwords_phrase_tree.items():
        fractions.append(len(head_properties["set"]) / (len(head_properties["list"])))
        sets.append(len(head_properties["set"]))

    score = np.sum(np.array(fractions)) * np.sum(np.array(sets)) / len(mentions) if len(mentions) > 1 else 1
    return float(format(score, '.3f'))


def pd_lea(input_clusters, output_clusters, mention_to_gold,dict_phrasing_diversity,cluster_type):
    num, den = 0, 0
    mention_pd_dict = {}
    for c in input_clusters:
        phrasing_diversity = 0 #Default for singletons
        if len(c) == 1:
            #Add only mention with pd=0
            mention_pd_dict[c[0].id]=0
            all_links = 1
            if c[0] in mention_to_gold and len(
                    output_clusters[mention_to_gold[c[0]]]) == 1:
                common_links = 1
            else:
                common_links = 0
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            mention_list = []
            coreference_chain_list = []
            label_list = []
            mention_id_list = []
            for i, m in enumerate(c):
                mention_list.append(dict_phrasing_diversity[m.id])
                coreference_chain_list.append(m.coreference_chain)
                mention_id_list.append(m.id)
                if m in mention_to_gold:
                    label_list.append(mention_to_gold[m])
                    for m2 in c[i + 1:]:
                        if m2 in mention_to_gold and mention_to_gold[
                                m] == mention_to_gold[m2]:
                            common_links += 1
                        # else:
                        #    print('!! ', m2, '--', m2.get_span(), ' ',
                        #           m2.min_spans, ' ', mention_to_gold[m], ' ',
                        #           mention_to_gold[m2], ' ' ,
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m]]], ' -- ',
                        #           [str(s) for s in output_clusters[
                        #               mention_to_gold[m2]]])
                else:
                    label_list.append(-1)
                            
            phrasing_diversity = phrasing_diversity_calc(mention_list)
            #Add mentions to the dictionary with phrasing diversity
            for id,cluster_label,coref_chain in zip(mention_id_list,label_list,coreference_chain_list):
                mention_pd_dict[id]=[phrasing_diversity,cluster_label,coref_chain]
            
            #dict to dataframe
            mention_pd_df = pd.DataFrame.from_dict(mention_pd_dict,orient='index')
            mention_pd_df = mention_pd_df.explode(0).reset_index()
            if cluster_type=='gold':
                mention_pd_df.columns=['mention_id','pd_'+cluster_type,'predicted_label',"coref_chain"]
            else:
                mention_pd_df.columns=['mention_id','pd_'+cluster_type,'gold_label',"coref_chain"]
                mention_pd_df.drop(columns=['coref_chain'], inplace=True)

        num += phrasing_diversity * common_links / float(all_links) #This len(c) in the numerator must be the importance factor
        den += phrasing_diversity

    return num, den, mention_pd_df