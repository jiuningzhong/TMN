# http://localhost:8888/notebooks/Documents/GitHub/topic-model-tutorial/1%20-%20Text%20Preprocessing.ipynb
# 1 - Text Preprocessing

import csv
from sklearn.feature_extraction.text import CountVectorizer
import gensim

raw_documents = []
snippets = []

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
        # keep a short snippet of up to 100 characters as a title for each article
        snippets.append( text[0:min(len(text), 100)] )

print("Read %d raw text documents" % len(raw_documents))

print(raw_documents)

custom_stop_words = []
with open( "stopwords.txt", "r" ) as fin:
    for line in fin.readlines():
        custom_stop_words.append( line.strip() )
# note that we need to make it hashable
print("Stopword list has %d entries" % len(custom_stop_words) )

# use a custom stopwords list, set the minimum term-document frequency to 20
vectorizer = CountVectorizer(stop_words = custom_stop_words, min_df = 20)
A = vectorizer.fit_transform(raw_documents)
print( "Created %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )

terms = vectorizer.get_feature_names()
print("Vocabulary has %d distinct terms" % len(terms))

from sklearn.externals import joblib
joblib.dump((A,terms,snippets), "articles-raw.pkl")

from sklearn.feature_extraction.text import TfidfVectorizer
# we can pass in the same preprocessing parameters
vectorizer = TfidfVectorizer(use_idf = True,
                             stop_words=custom_stop_words,
                             min_df = 15, #20
                             #max_df = 200, #20
                             ngram_range=(1,1),
                             analyzer = 'word',
                             lowercase = True,
                             token_pattern = '[a-zA-Z0-9]{3,}',
                             max_features = 2000,
                             sublinear_tf = True
                            )
#vectorizer = TfidfVectorizer(use_idf = True, ngram_range=(1,4), stop_words = custom_stop_words, # ngram_range=(1,6)
#                                        analyzer = 'word',
#                                        # min_df = 3,  # minimum required occurences of a word
#                                        # min_df = 3
#                                        lowercase = True,  # convert all words to lowercase
#                                        token_pattern = '[a-zA-Z0-9]{3,}',  # num chars > 3
#                                        max_features = 8000,
#                                        sublinear_tf = True,
# max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
#)

A = vectorizer.fit_transform(raw_documents)
print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )

# extract the resulting vocabulary
terms = vectorizer.get_feature_names()
print("Vocabulary has %d distinct terms" % len(terms))

import operator
def rank_terms( A, terms ):
    # get the sums over each column
    sums = A.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

ranking = rank_terms( A, terms )
for i, pair in enumerate( ranking[0:20] ):
    print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )

joblib.dump((A,terms,snippets), "articles-tfidf.pkl")
