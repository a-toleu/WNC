#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gensim
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import pickle
import random 
import json
from tqdm import tqdm_notebook as tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from gensim.models.coherencemodel import CoherenceModel
import numba as nb
import warnings
warnings.filterwarnings('ignore')

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    #pickle.dump(obj,open(name + '.pkl', 'wb'))
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=4)


# In[3]:

# bow_corpus = load_obj('bow_corpus')

# data_df = pd.read_csv('20news_train_df.csv')
# # In[34]:


# tokens_text = []
# for doc in data_df['News'].values.astype('U'):
#     for sent in sent_tokenize(doc):
#         words = word_tokenize(sent)
#         tokens_text.append(words)


# from gensim.corpora import Dictionary
# dictionary = Dictionary(tokens_text)
# #|bow_corpus = dictionary.doc2bow(tokens_text)

def Umass_coherence(topical_words,n_topword,bow_corpus,dictionary):
    topics_ = []
    for wt in topical_words:
        topics_.append(wt[:n_topword])
        #topics_.append(wt)
        
    # Compute Coherence Score
    cm = CoherenceModel(topics = topics_,  corpus = bow_corpus, dictionary = dictionary, 
                        coherence='u_mass', topn=n_topword)
    score = cm.get_coherence()
    return score

def Uci_coherence(topical_words,n_topword,tokens_text,dictionary):
    topics_ = []
    for wt in topical_words:
        topics_.append(wt[:n_topword])
        #topics_.append(wt)
        
    # Compute Coherence Score
    cm = CoherenceModel(topics = topics_,  texts = tokens_text, dictionary = dictionary, 
                        coherence='c_uci', topn=n_topword)
    score = cm.get_coherence()
    return score

def cv_coherence(topical_words,n_topword,tokens_text,dictionary):
    topics_ = []
    for wt in topical_words:
        topics_.append(wt[:n_topword])
        #topics_.append(wt[:])
        
    # Compute Coherence Score
    cm = CoherenceModel(topics = topics_,  texts = tokens_text, dictionary = dictionary, 
                        coherence='c_v', topn=n_topword)
    score = cm.get_coherence()
    return score





