"""
   Create corpus for further use.
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
import pathlib

pathlib.Path('./tmp').mkdir(exist_ok=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)
corpus = UC.UbuntuCorpus("data/dialogs_4.txt", "tmp/dialogs4.dict")

corpora.MmCorpus.serialize('tmp/dialogs4-corpus.mm', corpus)
