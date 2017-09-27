#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: saheli Saha 06
"""

import pandas as pd
import pprint, pickle
import numpy as np
import sklearn.utils
    
pkl_file = open('fintech_cleaned.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

dataset = data.copy()
dataset_academia = dataset.loc[dataset['stakeholder'] == 'Academia']
dataset_ambiguous = dataset.loc[dataset['stakeholder'] == 'Ambiguous']
dataset_bus = dataset.loc[dataset['stakeholder'] == 'Business Representatives']
dataset_companies = dataset.loc[dataset['stakeholder'] == 'Companies']
dataset_corporate = dataset.loc[dataset['stakeholder'] == 'Corporate Interest Groups']
dataset_expert = dataset.loc[dataset['stakeholder'] == 'Expert Institutions']
dataset_individual = dataset.loc[dataset['stakeholder'] == 'Individual Experts']
dataset_Non_Corporate = dataset.loc[dataset['stakeholder'] == 'Non-Corporate Interests']
dataset_policy = dataset.loc[dataset['stakeholder'] == 'Policymakers']
dataset_private = dataset.loc[dataset['stakeholder'] == 'Private Person']
dataset_media = dataset.loc[dataset['stakeholder'] == 'Media']

# Code to get the data for Academia group only
dataset_academia = sklearn.utils.shuffle(dataset_academia)
# Code to get the data for Ambiguous group only
#dataset_ambiguous = sklearn.utils.shuffle(dataset_ambiguous)
# Code to get the data for Business Representative group only
#dataset_bus = sklearn.utils.shuffle(dataset_bus)
# Code to get the data for Companies group only
#dataset_companies = sklearn.utils.shuffle(dataset_companies)
# Code to get the data for Corporate Interests group only
#dataset_corporate = sklearn.utils.shuffle(dataset_corporate)
# Code to get the data for Expert Institutions group only
#dataset_expert = sklearn.utils.shuffle(dataset_expert)
# Code to get the data for Individual Experts group only
#dataset_individual = sklearn.utils.shuffle(dataset_individual)
# Code to get the data for Non Corporate Interests group only
#dataset_Non_Corporate = sklearn.utils.shuffle(dataset_Non_Corporate)
# Code to get the data for Policymakers group only
#dataset_policy = sklearn.utils.shuffle(dataset_policy)
# Code to get the data for Private Person group only
#dataset_private = sklearn.utils.shuffle(dataset_private)
# Code to get the data for Media group only
#dataset_media = sklearn.utils.shuffle(dataset_media)

# Shuffled and rearranging for indexing
tweet = dataset_media['tweet']
tweet = np.array(tweet)
tweet = pd.DataFrame(tweet)

user = dataset_media['user']
user = np.array(user)
user = pd.DataFrame(user)

date = dataset_media['date']
date = np.array(date)
date = pd.DataFrame(date)

description = dataset_media['description']
description = np.array(description)
description = pd.DataFrame(description)

location = dataset_media['location']
location = np.array(location)
location = pd.DataFrame(location)

followers = dataset_media['followers']
followers = np.array(followers)
followers = pd.DataFrame(followers)

stakeholder = dataset_media['stakeholder']
stakeholder = np.array(stakeholder)
stakeholder = pd.DataFrame(stakeholder)

new_data = pd.concat([tweet, user, date, description, location, followers, stakeholder], axis = 1)
new_data.columns = ['tweet', 'user', 'date', 'description', 'location', 'followers', 'stakeholder' ]

new_data_copy = new_data.copy()

#Formation of corpus after cleaning the tweets

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
for i in range(len(new_data)):
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

#Formation of corpus for location after cleaning the location attribute
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
    
#Bigrams  for tweets  
from nltk import bigrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
bigram_measures= BigramAssocMeasures() 
terms_bigram = bigrams(corpus)
for i in range(len(corpus)):
    for word in corpus[i]:
        tweet = corpus[i]
        tweet = tweet.split()
        finder = BigramCollocationFinder.from_words(tweet)
# This gets the top 10 bigrams according to PMI
        bi_terms = finder.nbest(bigram_measures.pmi,10)
print(bi_terms)

#Bigram for location
bigram_measures= BigramAssocMeasures() 
terms_bigram = bigrams(corpus_location)
for i in range(len(corpus_location)):
    for word in corpus_location[i]:
        tweet = corpus_location[i]
        tweet = tweet.split()
        finder = BigramCollocationFinder.from_words(tweet)
# This gets the top 10 bigrams according to PMI
        bi_terms = finder.nbest(bigram_measures.pmi,10)
print(bi_terms)

#Co- Occurance Checking for tweets
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
# Creating 20 most common used words                    
com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:20])

#Co- Occurance Checking for location
import operator 
from collections import Counter
from collections import defaultdict
com = defaultdict(lambda : defaultdict(int))
count_all = Counter()
for i in range(len(corpus_location)):
    for word in corpus_location[i]:
        tweet = corpus_location[i]
        tweet = tweet.split()
        # Create a list with all the terms
        terms_all = [term for term in tweet]
        # Build co-occurrence matrix
        for i in range(len(terms_all)-1):            
            for j in range(i+1, len(terms_all)):
                w1, w2 = sorted([terms_all[i], terms_all[j]])                
                if w1 != w2:
                    com[w1][w2] += 1
# Creating 20 most common used words                    
com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print(terms_max[:20])

#Each Words Frequency for tweets
all_words = []
sorted_freq_word = defaultdict(lambda : defaultdict(int))

for i in range(len(corpus)):
    tweet = corpus[i]
    tweet = tweet.split()
    for w in tweet:
        all_words.append(w)
        
all_words_freq = nltk.FreqDist(all_words)

#Each Words Frequency for Location
all_words = []
sorted_freq_word = defaultdict(lambda : defaultdict(int))

for i in range(len(corpus_location)):
    tweet = corpus_location[i]
    tweet = tweet.split()
    for w in tweet:
        all_words.append(w)
        
all_words_freq = nltk.FreqDist(all_words)

sorted_freq_word = all_words_freq.most_common()
print(sorted_freq_word[:20])

#Sentiment Analysis for tweets
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
avg_pos = avg_pos/len(new_data)
avg_neg = avg_neg/len(new_data)
print (avg_neg, avg_pos)
print "Top 20 words Sentiments"
print (sentiment_orientation[:20])

#Topic Modelling

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


no_features = 2000 # Decided the value after tuning to get the best result
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20
no_topics_range = [20, 30, 40, 50, 60] # No of topics range taken for parameter tuning to get the best result
no_top_words = 30
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




