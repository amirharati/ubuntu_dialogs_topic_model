"""
  A wrapper for clustering algorithm.
"""
import numpy as np
import re
from gensim import corpora, models, similarities
from nltk import cluster
from nltk.cluster import euclidean_distance, cosine_distance


class VecCluster:
    def __init__(self, n_clusters, dict_file):
        self.n_clusters = n_clusters
        self.pattern = re.compile('[&!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]')
        self.dictionary = corpora.Dictionary.load_from_text(dict_file)

    def predict(self, feat):
        p = self.clusterer.classify(feat)
        return p, self.cluster_names[p]

    def cluster(self, feats_file, num_training, docs_file):
        """ feats_file: file contains features
            num_training: number of random data point from file to use for traninig.
            Because of limitted computation power we have to train our clusters on subset of data.
            doc_files: file contains all original data with their id.
        """
        lines = [line.strip() for line in open(feats_file)]
        feats = []
        for line in lines:
            parts = line.split()
            v = [float(x) for x in parts[1].split(",")]
            feats.append(v)
        feats = np.array(feats)

        # use nltk clustering because it has cosine distance
        self.clusterer = cluster.KMeansClusterer(self.n_clusters, cosine_distance, repeats=3, avoid_empty_clusters=True)
        # randomly select 10000 data points
        self.clusters = self.clusterer.cluster(feats[np.random.choice(feats.shape[0], num_training, replace=False), :], False, True)
        P = []
        for i in range(feats.shape[0]):
            p = self.clusterer.classify(feats[i, :])
            P.append(p)
        # load the docs
        lines = [line.strip() for line in open(docs_file)]
        docs = []
        for line in lines:
            docs.append(line)

        clean_docs = []
        # pass the docs through pipeline to remove stopwords etc
        for doc in docs:
            v = []
            words = self.pattern.sub(" ", doc.lower()).split()
            for word in words:
                if word in self.dictionary.token2id:
                    v.append(word)
            #if (len(v) > 0):
            clean_docs.append(v)

        # find what are words corresponding to each cluster
        words_cluster = {}
        cluster_counter = {}
        for itr in range(len(clean_docs)):
            cn = P[itr]
            words = clean_docs[itr]
            if cn not in cluster_counter:
                cluster_counter[cn] = 1
            cluster_counter[cn] += 1

            if cn not in words_cluster:
                words_cluster[cn] = {}
            for w in words:
                if w not in words_cluster[cn]:
                    words_cluster[cn][w] = 1
                else:
                    words_cluster[cn][w] += 1

        # find a albel for each cluster
        cluster2word = {}
        for cn in words_cluster:
            sorted_words = []
            for w in sorted(words_cluster[cn], key=words_cluster[cn].get, reverse=True):
                sorted_words.append(w)
            cluster2word[cn] = sorted_words

        self.main_topics_id = []
        for i in sorted(cluster_counter, key=cluster_counter.get, reverse=True):
            self.main_topics_id.append(i)

        #print(main_topics_id)
        self.cluster_names = {}
        for ci in cluster2word:
            v = cluster2word[ci][:4]
            #print(ci, " --- ", v)
            self.cluster_names[ci] = "-".join(v)
        #print(cluster2word)

        print("TOP TOPICS")
        for i in range(10):
            tid = self.main_topics_id[i]
            print(tid, " --- ", self.cluster_names[tid])
