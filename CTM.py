#!/usr/bin/env python
# coding: utf-8

# In[116]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances

import re
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from community import community_louvain
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
from tm_eval import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist 


# In[123]:


class CTM:
    
    def __init__(self):
        self.topics = []
        self.docs = []
    
    
    def getvocab_and_vectorized(self, docs, stopwords, optimized=0, njobs = 20):

        if optimized == 0:
            vocab = []
            vectorizer = CountVectorizer(stop_words=stopwords, min_df=10,ngram_range=(1,1))
            X_cooc = vectorizer.fit_transform(docs)
            for x in vectorizer.get_feature_names_out():
                vocab.append(x)

            coocX = 1 - pairwise_distances(X_cooc.T, metric = 'cosine', n_jobs = njobs) # cooc-based

            wtoi = {w:i for i, w in enumerate(vocab)}
            itow = {i:w for i, w in enumerate(vocab)}

            return vocab, coocX, wtoi, itow
        else:
            tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df = 1000, min_df = 5)
            X_tfidf = tfidf_vectorizer.fit_transform(docs)

            vocab_tfidf = []
            vocab = []
            print("extracting optimized vocabulary...")
            for i in range(len(docs)):#len(news_train_df['News'])

                df = pd.DataFrame(X_tfidf[i].T.todense(), index= tfidf_vectorizer.get_feature_names_out(), columns=["tfidf"])

                sorted_tfidf = df.sort_values(by=["tfidf"],ascending=False)
                sorted_filltered_ws = sorted_tfidf[0:10]

                for j,w in enumerate(sorted_filltered_ws.index):
                    if w not in vocab:
                        vocab.append(w)
                        vocab_tfidf.append((w,sorted_filltered_ws['tfidf'][j]))

            print("pariwise calculation...")
            tfid_vec = TfidfVectorizer(use_idf=True, max_df = 1000, min_df = 5, tokenizer=lambda x: str(x).split(),vocabulary=vocab)
            X_tfidf = tfid_vec.fit_transform(docs)

            tfidfX = 1 - pairwise_distances(X_tfidf.T, metric = 'cosine', n_jobs = njobs)

            wtoi = {w:i for i, w in enumerate(vocab)}
            itow = {i:w for i, w in enumerate(vocab)}

            return vocab,tfidfX, wtoi, itow
        
    def similar_words(self, w1,model,tpn=100):
        p1 = wtoi[w1]
        scores = {}
        for w,i in wtoi.items():
            if w1 != w:
                #sim = cosine_similarity(model[p1],model[i])[0][0]
                sim = model[p1][i]
                scores[w] = sim 
        sorted_w = sorted(scores.items(),key=lambda x:x[1], reverse=True)
        return sorted_w[0:tpn]
    
    def words_distance(self, w1, w2, wvmodel, wtoi):
        sim = 0
        try:
            p1 = wtoi[w1]
            p2 = wtoi[w2]
            #sim = cosine_similarity(model[p1],model[p2])[0][0]
            sim = wvmodel[p1][p2]
        except KeyError:
            sim = 0

        return sim
    
    def w2v_projecting(self, vocab, wtoi, wvmodel):   
        embeds = []
        dim = wvmodel.shape[0]
        vec = np.zeros((dim,1))
        for i, word in enumerate(vocab):
            if word in wtoi:
                idx = wtoi[word]
                vec = wvmodel[idx].reshape(dim,1)
            embeds.append(vec.flatten())
        return embeds

    def build_network_and_clustering(self, vocab, wvmodel, resolution = 0.85, nodeOpti=50, ranking=1):
        G = nx.Graph()
        for i,n1 in enumerate(vocab[0:]):
            G.add_node(n1)
            queue = []
            for n2 in vocab[i+1:]:
                sim = self.words_distance(n1, n2, wvmodel, wtoi)

                if sim > 0.0:
                    queue.append((sim, n2))

            for sim, n2 in sorted(queue)[-1*nodeOpti:]:#-50
                v = {'w':sim}
                G.add_edge(n1,n2)
                G[n1][n2].update(v)

        partition = community_louvain.best_partition(G, resolution=resolution)

        #generate topics
        topics_unsorted = {}
        for k,v in partition.items():
            if v not in topics_unsorted:
                topics_unsorted[v] = [k]
            else:
                ws = topics_unsorted[v]
                ws.append(k)
                topics_unsorted[v] = ws

        #ranking by its contained words
        topic_sorted_size = sorted(topics_unsorted.items(), key=lambda x:len(x[1]),reverse=True)

        #get topics and ranking each word in side topics by node's degrees
        topics_sorted =[]
        for i,topic in topic_sorted_size:
            _topic = []
            for w in topic:
                _topic.append((w,len(G[w])))
            if ranking == 1:
                topics_sorted.append([w for w,e in sorted(_topic,key=lambda x:x[1], reverse=True)])
            else:
                topics_sorted.append([w for w,e in _topic])
        return G, topics_sorted

    def kMeans_tm(self, vocab, wtoi, wvmodel, n, initA='random', max_iter=100, topM=10):
        
        X = np.array(self.w2v_projecting(vocab, wtoi, wvmodel))
        kmeans = KMeans(n_clusters=n, init=initA, max_iter=max_iter, n_init=1, algorithm='auto') #cluster_number, random, k-means++,
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        centroids = kmeans.cluster_centers_

        
        #get topics
        clusters = []
        clusters_vec = []
        for i in range(0, n):
            tpl = []
            tpl_vec = []
            for j, word in enumerate(vocab):
                if i == y_kmeans[j]:
                    tpl.append(word)
                    tpl_vec.append(X[j])
            clusters.append(tpl)
            clusters_vec.append(tpl_vec)
            
        #calculate weights of topics
        topic2word = []
        for i,topic in enumerate(clusters):
            topic2word_prob = np.ones(1)
            if len(topic) > 1:
                topic2word_prob = euclidean_distances([centroids[i]],clusters_vec[i])[0] #,cosine_similarity          
            topic2word.append(list(zip(topic, softmax(np.transpose(topic2word_prob)))))
            
        #ranking words inside each topic
        topics = []
        for i, cluster in enumerate(clusters):
            topic = [x[0] for x in sorted(topic2word[i], key=lambda x: x[1], reverse=True)]
            if len(topic) > 0:
                #print('topic', i, ':', ', '.join(topic[0:20]), '\n')
                topics.append(topic)
        
        return topics, centroids


# In[ ]:


#Load


# In[68]:


from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords


# In[69]:


stopwords = stopwords.words('english')
newsgroups_test = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
docs = newsgroups_test['data']


# In[ ]:





# In[70]:


vocab,w2vmodel, wtoi, itow = ctm.getvocab_and_vectorized(docs,stopwords,optimized=0,njobs=1)


# In[71]:


ctm.similar_words("apple", w2vmodel)


# In[56]:


topics, centroids = ctm.kMeans_tm(vocab=vocab, wtoi=wtoi, wvmodel=w2vmodel, n=20, initA="random", max_iter=100, topM=10)


# In[82]:


G, topics = ctm.build_network_and_clustering(vocab, w2vmodel, resolution = 0.87, nodeOpti=50, ranking=1)


# In[83]:


len(topics)


# In[84]:


len(vocab)


# In[103]:


from gensim.corpora import Dictionary

tokens_text = []
for doc in docs:
    for sent in sent_tokenize(doc):
        words = word_tokenize(sent)
        if len(words) > 0:
            tokens_text.append(words)



dictionary = Dictionary(tokens_text)
bow_corpus = [dictionary.doc2bow(sent, allow_update=True) for sent in tokens_text]


# In[119]:


umass = []
uci = []
cv = []
tpns = np.arange(10,15,5)
for tpn in tpns:
    tc1 = Umass_coherence(topics, tpn, bow_corpus, dictionary)
    tc2 = Uci_coherence(topics, tpn, tokens_text, dictionary)
    tc3 = cv_coherence(topics, tpn, tokens_text,dictionary)

    umass.append(tc1)
    uci.append(tc2)
    cv.append(tc3)


# In[120]:


eval_tc = {'umass':umass,'uci':uci,'cv':cv}


# In[121]:


eval_tc


# In[ ]:




