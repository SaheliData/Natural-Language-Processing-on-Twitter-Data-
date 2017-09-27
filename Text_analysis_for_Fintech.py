#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: saheli06
"""

import pandas as pd
import pprint, pickle
import numpy as np
import sklearn.utils
    
pkl_file = open('fintech_cleaned.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

dataset = data.copy()

# There are many missing observations with index values, hence, rearranging for indexing
tweet = dataset['tweet']
tweet = np.array(tweet)
tweet = pd.DataFrame(tweet)

user = dataset['user']
user = np.array(user)
user = pd.DataFrame(user)

date = dataset['date']
date = np.array(date)
date = pd.DataFrame(date)

description = dataset['description']
description = np.array(description)
description = pd.DataFrame(description)

location = dataset['location']
location = np.array(location)
location = pd.DataFrame(location)

followers = dataset['followers']
followers = np.array(followers)
followers = pd.DataFrame(followers)

stakeholder = dataset['stakeholder']
stakeholder = np.array(stakeholder)
stakeholder = pd.DataFrame(stakeholder)

new_data = pd.concat([tweet, user, date, description, location, followers, stakeholder], axis = 1)
new_data.columns = ['tweet', 'user', 'date', 'description', 'location', 'followers', 'stakeholder' ]
new_data_copy = new_data.copy()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import unicodedata
import string
tokenizer = RegexpTokenizer(r'\w+')
punctuation = list(string.punctuation)
corpus = []

# Formation of corpus after cleaning the tweets   
for i in range(0,500):
    review = re.sub('[^a-zA-Z]', ' ', new_data['tweet'][i])
    review = review.lower()
    unicodedata.normalize('NFKD', review).encode('ascii','ignore')
    review = review.encode('utf8')
    r = review
    review = tokenizer.tokenize(review)
    review = [str(x) for x in review]
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english') + punctuation + ['rt', 'https','via', 'co'])]
    review = [str(x) for x in review]
    review = ' '.join(review)
    corpus.append(review)

# Formation of corpus after cleaning the Location Attribute  
corpus_location = []
for i in range(len(new_data)):
    lc = re.sub('[^a-zA-Z]', ' ', new_data['location'][i])
    lc = lc.lower()
    unicodedata.normalize('NFKD', lc).encode('ascii','ignore')
    lc = lc.encode('utf8')
    l = lc
    lc = tokenizer.tokenize(lc)
    lc = [str(x) for x in lc]
    #ps = PorterStemmer()
    lc = [word for word in lc if not word in set(stopwords.words('english') + punctuation + ['rt', 'https','via', '#', 'co'])]
    lc = [str(x) for x in lc]
    lc = ' '.join(lc)
    corpus_location.append(lc)

# Formation of corpus after cleaning the Description Attribute
corpus_desc = []
for i in range(len(new_data)):
    desc = re.sub('[^a-zA-Z]', ' ', new_data['description'][i])
    desc = desc.lower()
    unicodedata.normalize('NFKD', desc).encode('ascii','ignore')
    desc = desc.encode('utf8')
    d = desc
    desc = tokenizer.tokenize(desc)
    desc = [str(x) for x in desc]
    #ps = PorterStemmer()
    desc = [word for word in desc if not word in set(stopwords.words('english') + punctuation + ['rt', 'https','via', 'co'])]
    desc = [str(x) for x in desc]
    desc = ' '.join(desc)
    corpus_desc.append(desc)

#Bigrams for tweets    
from nltk import bigrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
bigram_measures= BigramAssocMeasures() 
terms_bigram = bigrams(corpus)
for i in range(len(corpus)):
    for word in corpus[i]:
        tweet = corpus[i]
        tweet = tweet.split()
        finder = BigramCollocationFinder.from_words(tweet)
# This gets the top 20 bigrams according to PMI
        bi_terms = finder.nbest(bigram_measures.pmi,20)
print(bi_terms)

# Co- Occurance matrix for tweets
import operator 
from collections import Counter
from collections import defaultdict
com = defaultdict(lambda : defaultdict(int))
count_all = Counter()
for i in range(len(corpus)):
    for word in corpus[i]:
        tweet = corpus[i]
        tweet = tweet.split()
        # Create a list with all the terms
        terms_all = [term for term in tweet]
        # Build co-occurrence matrix
        for i in range(len(terms_all)-1):            
            for j in range(i+1, len(terms_all)):
                w1, w2 = sorted([terms_all[i], terms_all[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
# Creating 50 most common used words                    
com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:50])

# Co Occurance matrix for locations
com_lo = defaultdict(lambda : defaultdict(int))
count_all = Counter()
for i in range(len(corpus_location)):
    for word in corpus_location[i]:
        lo = corpus_location[i]
        lo = lo.split()
        # Create a list with all the terms
        terms_all = [term for term in lo]
        # Build co-occurrence matrix
        for i in range(len(terms_all)-1):            
            for j in range(i+1, len(terms_all)):
                w1, w2 = sorted([terms_all[i], terms_all[j]])                
                if w1 != w2:
                    com_lo[w1][w2] += 1
# Creating 50 most common used words                    
com_max_lo = []
# For each term, look for the most common co-occurrent terms
for t1 in com_lo:
    t1_max_terms = sorted(com_lo[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max_lo.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max_lo = sorted(com_max_lo, key=operator.itemgetter(1), reverse=True)
print(terms_max_lo[:50])

# Co Occurance matrix for description
import operator 
from collections import Counter
from collections import defaultdict
com_desc = defaultdict(lambda : defaultdict(int))
count_all = Counter()
for i in range(len(corpus_desc)):
    for word in corpus_desc[i]:
        de = corpus_desc[i]
        de = de.split()
        # Create a list with all the terms
        terms_all = [term for term in de]
        # Build co-occurrence matrix
        for i in range(len(terms_all)-1):            
            for j in range(i+1, len(terms_all)):
                w1, w2 = sorted([terms_all[i], terms_all[j]])                
                if w1 != w2:
                    com_desc[w1][w2] += 1
# Creating 50 most common used words                    
com_max_desc = []
# For each term, look for the most common co-occurrent terms
for t1 in com_desc:
    t1_max_terms = sorted(com_desc[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max_desc.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max_desc = sorted(com_max_desc, key=operator.itemgetter(1), reverse=True)
print(terms_max_desc[:50])

# Single Word Frequency for tweets
all_words = []
sorted_freq_word = defaultdict(lambda : defaultdict(int))

for i in range(len(corpus)):
    tweet = corpus[i]
    tweet = tweet.split()
    for w in tweet:
        all_words.append(w)
        
all_words_freq = nltk.FreqDist(all_words)

sorted_freq_word = all_words_freq.most_common()
print(sorted_freq_word[:50])

# Single Word Frequency for Location
all_words_loc = []
sorted_freq_word_loc = defaultdict(lambda : defaultdict(int))

for i in range(len(corpus_location)):
    lo = corpus_location[i]
    lo = lo.split()
    for w in lo:
        all_words_loc.append(w)
        
all_words_freq_lo = nltk.FreqDist(all_words_loc)

sorted_freq_word_loc = all_words_freq_lo.most_common()
print(sorted_freq_word_loc[:50])

# Single Word Frequency for Description
all_words_desc = []
sorted_freq_word_desc = defaultdict(lambda : defaultdict(int))

for i in range(len(corpus_desc)):
    de = corpus_desc[i]
    de = de.split()
    for w in de:
        all_words_desc.append(w)
        
all_words_freq_desc = nltk.FreqDist(all_words_desc)

sorted_freq_word_desc = all_words_freq_desc.most_common()
print(sorted_freq_word_desc[:50])

#Sentiment Analysis 
sentiment_orientation = []

from pattern.en import sentiment
for i in range(len(corpus)):
    tweet = corpus[i]
    sentiment_orientation.append(sentiment(tweet))

avg_pos = 0.00
avg_neg = 0.00
for n in range(len(sentiment_orientation)):
    avg_pos = avg_pos + sentiment_orientation[n][1]
    avg_neg = avg_neg + sentiment_orientation[n][0]
avg_pos = avg_pos/len(corpus)
avg_neg = avg_neg/len(corpus)

print (avg_neg, avg_pos)
print (sentiment_orientation[:30])

#Topic Modelling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


no_features = 2500 # Decided the value after tuning to get the best result 
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 30 
no_topics_range = [20, 30, 40, 50, 60] # No of topics range taken for parameter tuning to get the best result
no_top_words = 20
# Run NMF
#for n_clusters in no_topics_range:
#    print "Display NMF model for", n_clusters
#    nmf = NMF(n_components=n_clusters, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
#    display_topics(nmf, tfidf_feature_names, no_top_words)

# Run LDA
#for n_clusters in no_topics_range:
#    print "Display LDA model for", n_clusters
#    lda = LatentDirichletAllocation(n_topics=n_clusters, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
#    display_topics(lda, tf_feature_names, no_top_words)

# Run NMF and LDA
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)




