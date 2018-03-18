"""
   Create corpus for further use.
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
import pathlib
import pickle

pathlib.Path('./tmp').mkdir(exist_ok=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)
corpus = list(UC.UbuntuCorpus("data/dialogs_4.txt", "tmp/dialogs4.dict", tagged_document=True))

fo = open("tmp/dialogs4-tagged-corpus.pickle", 'wb')
pickle.dump(corpus, fo)
fo.close()
#for c in corpus:
#    print(c)
#corpora.MmCorpus.serialize('tmp/dialogs4-tagged-corpus.mm', corpus)
