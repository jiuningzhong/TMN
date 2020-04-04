import numpy as np
import gensim
import os, csv
import sys
from scipy import sparse
import pickle
import json
from gensim.parsing.preprocessing import STOPWORDS
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


class Dict(dict):
    def __missing__(self, key):
        return 0

# text_dict = Dict()
# complexity_dict = Dict()


# data_file = '../data/tmn/Train_all_types_hashes.csv'

# data_dir = os.path.dirname(data_file)
# with open(os.path.join(data_file), encoding='utf-8') as fin:
    # text = gensim.utils.to_unicode(fin.read(), 'latin1').strip()
    # print(text)

csv.register_dialect("hashes", delimiter="#")

# if __name__ == "__main__":
# d = os.getcwd()
# print(d)
# data_file = path.join(d, 'Train_all_types_hashes.csv')
# print(data_file)

data_file = '../data/tmn/Train_all_types_hashes.csv'

data_dir = os.path.dirname(data_file)

msgs = []
labels = []
label_dict = {}

with open(os.path.join(data_file), mode='r', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="#")
    line_count = 0
    for row in csv_reader:
        msg = list(gensim.utils.tokenize(row[0].strip(), lower=True))
        label = row[1]
        msgs.append(msg)
        # labels.append(label)

        if label not in label_dict:
            label_dict[label] = len(label_dict)
        labels.append(label_dict[label])

print(msgs)

'''
print(msgs[0])
print(labels[0])
print(msgs[1])
print(labels[1])
print(label_dict)

# build bigram model
# bigram_mdl = gensim.models.phrases.Phrases(msgs, min_count=1, threshold=2)

CUSTOM_FILTERS = [remove_stopwords, stem_text]
msgs = [preprocess_string(" ".join(doc), CUSTOM_FILTERS) for doc in msgs]

# apply bigram model on tokens
# bigrams = bigram_mdl[msgs]

pprint.pprint(list(msgs)[0])

'''

dictionary = gensim.corpora.Dictionary(msgs)

print(dictionary)

import copy
bow_dictionary = copy.deepcopy(dictionary)
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
print(len_1_words)
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
# bow_dictionary.filter_extremes(no_below=3, keep_n=None)
bow_dictionary.compactify()

def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []
    # build bow
    row = []
    col = []
    value = []
    row_id = 0
    m_labels = []

    for d_i, doc in enumerate(text_doc):
        # if len(bow_dictionary.doc2bow(doc)) < 3:    # filter too short
        #    continue
        for i, j in bow_dictionary.doc2bow(doc):
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1

        wids = list(map(seq_dictionary.token2id.get, doc))
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
    lens = list(map(len, seq_doc))
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))
    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels

# print(len(msgs))
print(bow_dictionary)
# print(labels)

seq_title, bow_title, label_title = get_wids(msgs, dictionary, bow_dictionary, labels)

# print(seq_title)
# print(bow_title)
# print(label_title)

# split data
indices = np.arange(len(seq_title))
np.random.shuffle(indices)
nb_test_samples = int(0.2 * len(seq_title))

seq_title = np.array(seq_title)[indices]
seq_title_train = seq_title[:-nb_test_samples]
seq_title_test = seq_title[-nb_test_samples:]
bow_title = bow_title.tocsr()
bow_title = bow_title[indices]
bow_title_train = bow_title[:-nb_test_samples]
bow_title_test = bow_title[-nb_test_samples:]
label_title = np.array(label_title)[indices]
label_title_train = label_title[:-nb_test_samples]
label_title_test = label_title[-nb_test_samples:]

# save
logging.info("save data...")
pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))
pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))
pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))
dictionary.save(os.path.join(data_dir, "dataDictSeq"))
bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("done!")