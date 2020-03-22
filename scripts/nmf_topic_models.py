# NMF Topic Models
# 1. Applying NMF
from sklearn.externals import joblib

(A,terms,snippets) = joblib.load( "articles-raw.pkl" ) # articles-tfidf.pkl
print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )

k = 30

# create the model
from sklearn import decomposition
model = decomposition.NMF( init="nndsvd", n_components=k )
# apply the model and extract the two factor matrices
W = model.fit_transform( A )
H = model.components_



# 2. Examining the Output
# NMF produces to factor matrices as its output: W and H.
# The W factor contains the document membership weights relative to each of the k topics.
# Each row corresponds to a single document, and each column correspond to a topic.
print(W.shape)

# round to 2 decimal places for display purposes
# For instance, for the first document, we see that it is strongly associated with one topic.
# However, each document can be potentially associated with multiple topics to different degrees.
W[0,:].round(2)

print(H.shape)

term_index = terms.index('light')
# round to 2 decimal places for display purposes
# The H factor contains the term weights relative to each of the k topics.
# In this case, each row corresponds to a topic, and each column corresponds to a unique term in the corpus vocabulary.
print(H[:,term_index].round(2))

# 3. Topic Descriptors
# The top ranked terms from the H factor for each topic can give us an insight into the content of that topic.
# This is often called the topic descriptor. Let's define a function that extracts the descriptor for a specified topic:
import numpy as np
def get_descriptor( terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
    return top_terms

descriptors = []
for topic_index in range(k):
    descriptors.append( get_descriptor( terms, H, topic_index, 10 ) )
    str_descriptor = ", ".join( descriptors[topic_index] )
    print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )
