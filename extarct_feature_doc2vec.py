"""
    pass the raw data through doc2vec model to obtain vec presentation.
    all create id2doc table.
"""
from gensim import corpora, models
import logging
import re
import data_prep as dp
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)

model = models.doc2vec.Doc2Vec.load("tmp/doc2vec.model")

raw_corpus = [line.strip() for line in open("data/dialogs_4.txt")]

pattern = re.compile('[&!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]')
dictionary = corpora.Dictionary.load_from_text('tmp/dialogs4.dict')

fo1 = open("tmp/id2doc.txt", "w")
fo2 = open("tmp/id2feat.txt", "w")
idc = 0
for doc in raw_corpus:
    v = []
    words = pattern.sub(" ", doc.lower()).split()
    for word in words:
        if word in dictionary.token2id:
            v.append(word)

    inferred_docvec = model.infer_vector(v)
    str_docvev = ",".join([str(x) for x in inferred_docvec])
    fo1.write(str(idc) + " " + doc + "\n")
    fo2.write(str(idc) + " " + str_docvev + "\n")
    idc += 1
fo1.close()
fo2.close()
