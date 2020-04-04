# http://localhost:8888/notebooks/Documents/GitHub/topic-model-tutorial/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
# 3 - Parameter Selection for NMF
from sklearn.externals import joblib

(A,terms,snippets) = joblib.load( "articles-tfidf.pkl" ) # articles-tfidf.pkl
print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )

kmin, kmax = 6, 50

# create the model
from sklearn import decomposition
topic_models = []
# try each value of k
for k in range(kmin,kmax+1):
    print("Applying NMF for k=%d ..." % k )
    # run NMF
    model = decomposition.NMF( init="nndsvd", n_components=k )
    W = model.fit_transform( A )
    H = model.components_
    # store for later
    topic_models.append( (k,W,H) )

import os.path
import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gensim
import os
import sys
from scipy import sparse
import pickle
import json
from gensim.parsing.preprocessing import STOPWORDS

raw_documents = []

file_path = '../data/Train_all_types_hashes.csv'
csv.register_dialect("hashes", delimiter="#")

with open(file_path, "r", encoding="utf8") as fin:
    csv_reader = csv.reader(fin, delimiter="#")

    for line in csv_reader:
        # text = line[0].strip()
        text = gensim.utils.to_unicode(line[0], 'latin1').strip()
        msg = list(gensim.utils.tokenize(text, lower=True))
        msg = ' '.join(msg)
        raw_documents.append( msg )


print("Read %d raw text documents" % len(raw_documents))

custom_stop_words = []
with open( "stopwords.txt", "r" ) as fin:
    for line in fin.readlines():
        custom_stop_words.append( line.strip().lower() )
# note that we need to make it hashable
print("Stopword list has %d entries" % len(custom_stop_words) )

import re
class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 2:
                    tokens.append( tok )
            yield tokens

import gensim
docgen = TokenGenerator( raw_documents, custom_stop_words )
# print(len(docgen.documents))
# the model has 219 dimensions, the minimum document-term frequency is 20
w2v_model = gensim.models.Word2Vec(docgen, sg=1,size=350, min_count=15) #

print( "Model has %d terms" % len(w2v_model.wv.vocab) )

w2v_model.save("w2v-model.bin")

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            if pair[0] not in w2v_model.wv.vocab or pair[1] not in  w2v_model.wv.vocab:
                continue
            pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

import numpy as np
def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

from itertools import combinations
k_values = []
coherences = []
for (k,W,H) in topic_models:
    # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append( get_descriptor( terms, H, topic_index, 10 ) )
    # Now calculate the coherence based on our Word2vec model
    k_values.append( k )
    coherences.append( calculate_coherence( w2v_model, term_rankings ) )
    print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

fig = plt.figure(figsize=(13,7))
# create the line plot
ax = plt.plot( k_values, coherences )
plt.xticks(k_values)
plt.xlabel("Number of Topics")
plt.ylabel("Mean Coherence")
# add the points
plt.scatter( k_values, coherences, s=120)
# find and annotate the maximum point on the plot
ymax = max(coherences)
xpos = coherences.index(ymax)
best_k = k_values[xpos]
plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
# show the plot
plt.show()

k = best_k
# get the model that we generated earlier.
W = topic_models[k-kmin][1]
H = topic_models[k-kmin][2]

for topic_index in range(k):
    descriptor = get_descriptor( terms, H, topic_index, 10 )
    str_descriptor = ", ".join( descriptor )
    print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )

