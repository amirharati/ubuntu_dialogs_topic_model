from gensim import corpora, models, similarities
import logging
import re
import data_prep as dp
import sys
import multiprocessing
import pickle
import UbuntuCorpus as UC

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)

cores = multiprocessing.cpu_count() - 1

#tagged_docs = models.doc2vec.TaggedLineDocument("data/dialogs_4.txt")

fo = open("tmp/dialogs4-tagged-corpus.pickle", 'rb')
tagged_docs = pickle.load(fo)
fo.close()
#tagged_docs = UC.UbuntuCorpus("data/dialogs_4.txt", "tmp/dialogs4.dict", tagged_document=True)
model = models.doc2vec.Doc2Vec(dm=0, alpha=0.025, dm_mean=1, size=10, window=10, negative=5, hs=0, min_count=2, workers=cores)  # use fixed learning rate
model.build_vocab(tagged_docs)
for epoch in range(10):
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay


model.save("tmp/doc2vec.model")
