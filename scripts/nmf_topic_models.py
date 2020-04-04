# http://localhost:8888/notebooks/Documents/GitHub/topic-model-tutorial/2%20-%20NMF%20Topic%20Models.ipynb
# 2 - NMF Topic Models

#%% md

# NMF Topic Models

#%% md
#%% md

### Applying NMF

#%% md
#%%

from sklearn.externals import joblib
(A,terms,snippets) = joblib.load( "articles-tfidf.pkl" ) # articles-tfidf.pkl
print( "Loaded %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )

#%% md
#%%

k = 50

#%% md
#%%

# create the model
from sklearn import decomposition
model = decomposition.NMF( init="nndsvd", n_components=k ) 
# apply the model and extract the two factor matrices
W = model.fit_transform( A )
H = model.components_

#%% md

### Examining the Output

#%% md
#%%

W.shape

#%% md
#%%

# round to 2 decimal places for display purposes
W[0,:].round(2)

#%% md
#%%

H.shape

#%% md
#%%

term_index = terms.index('light')
# round to 2 decimal places for display purposes
H[:,term_index].round(2)

#%% md

### Topic Descriptors

#%% md
#%%

import numpy as np
def get_descriptor( terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
    return top_terms

#%% md
#%%

descriptors = []
for topic_index in range(k):
    descriptors.append( get_descriptor( terms, H, topic_index, 10 ) )
    str_descriptor = ", ".join( descriptors[topic_index] )
    print("Topic %02d: %s" % ( topic_index+1, str_descriptor ) )

#%% md
#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

def plot_top_term_weights( terms, H, topic_index, top ):
    # get the top terms and their weights
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    # note we reverse the ordering for the plot
    top_terms.reverse()
    top_weights.reverse()
    # create the plot
    fig = plt.figure(figsize=(13,8))
    # add the horizontal bar chart
    ypos = np.arange(top)
    ax = plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)
    plt.xlabel("Term Weight",fontsize=14)
    plt.tight_layout()
    plt.show()

plot_top_term_weights( terms, H, 6, 15 )

#%% md

### Most Relevant Documents


def get_top_snippets( all_snippets, W, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( W[:,topic_index] )[::-1]
    # now get the snippets corresponding to the top-ranked indices
    top_snippets = []
    for doc_index in top_indices[0:top]:
        top_snippets.append( all_snippets[doc_index] )
    return top_snippets

topic_snippets = get_top_snippets( snippets, W, 0, 10 )
for i, snippet in enumerate(topic_snippets):
    print("%02d. %s" % ( (i+1), snippet ) )


topic_snippets = get_top_snippets( snippets, W, 1, 10 )
for i, snippet in enumerate(topic_snippets):
    print("%02d. %s" % ( (i+1), snippet ) )

#%% md

### Exporting the Results



joblib.dump((W,H,terms,snippets), "articles-model-nmf-k%02d.pkl" % k) 

#%%



#%%


