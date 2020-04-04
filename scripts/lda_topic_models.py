# Importing modules
import pandas as pd
import os
from os import path
# Load the regular expression library
import re
# Import the wordcloud library
from wordcloud import WordCloud

# print(os.getcwd())

# Read data into papers
papers = pd.read_csv('../data/Train_all_types_comma.csv')

# Print head
print(papers.head())

# Remove punctuation
papers['paper_text_processed'] = papers['text'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

# Print out the first rows of papers
papers['paper_text_processed'].head()


# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
wordcloud.to_image()

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Helper function
def plot_10_most_common_words(count_data, tfidf_vectorizer):
    import matplotlib.pyplot as plt
    words = tfidf_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# Initialise the count vectorizer with the English stop words
# count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
# count_data = count_vectorizer.fit_transform(papers['paper_text_processed'])

tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,5), stop_words='english', # ngram_range=(1,6)
                                    analyzer = 'word',
                                    min_df = 3,  # minimum required occurences of a word
                                    # min_df = 3
                                    lowercase = True,  # convert all words to lowercase
                                    token_pattern = '[a-zA-Z0-9]{3,}',  # num chars > 3
                                    max_features = 5000,
                                    sublinear_tf = True,
# max number of unique words. Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus
)

count_data = tfidf_vectorizer.fit_transform(papers['paper_text_processed'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, tfidf_vectorizer)

import warnings

warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# Tweak the two parameters below (use int values below 15)
number_topics = 15
number_words = 10

# Create and fit the LDA model
lda = LDA(n_components=number_topics)
lda.fit(count_data)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, tfidf_vectorizer, number_words)


from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis

# Visualize the topics
# pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('./ldavis_prepared_' + str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, tfidf_vectorizer)

    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_' + str(number_topics) + '.html')

LDAvis_prepared