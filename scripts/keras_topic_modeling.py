from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout
from keras.optimizers import Adam
import re, string
import numpy as np
import en_core_web_sm
import warnings
from IPython.core.interactiveshell import InteractiveShell
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import csv

if __name__ == "__main__":
    # Cleaning and Tokenizing the Text
    df_questions = pd.read_csv('../data/Train_all_types_comma.csv', encoding='utf-8')

    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(df_questions.text)
    # print(df_questions.text)
    sequences = tokenizer.texts_to_sequences(df_questions.text)
    print(len(sequences))
    x = pad_sequences(sequences, maxlen=180)

    # try LightGBM, Support Vector Machines or Logistic Regression with n-grams or tf-idf input features.
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(df_questions.Complexity_level)
    y = multilabel_binarizer.classes_

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

    # binary representations which mainly contain zeros and are high-dimensional.
    # Word embeddings on the other hand are low dimensional as they represent tokens
    # as dense floating point vectors and thus pack more information into fewer dimensions.
    #
    # Learn word embeddings together with the weights of the neural network
    # Load pretrained word embeddings which were precomputed as part of a different machine learning task.
    #
    # simple baseline
    maxlen = 180
    max_words = 5000
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(df_questions.Text)



