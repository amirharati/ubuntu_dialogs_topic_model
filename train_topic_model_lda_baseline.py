"""
  train baseline LDA model.
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)

dictionary = corpora.Dictionary.load_from_text('tmp/dialogs4.dict')
corpus = corpora.MmCorpus('tmp/dialogs4-corpus.mm')

# compute tfidf
tfidf = models.TfidfModel(corpus)

# convert the corpus to tfidf representation
corpus_tfidf = tfidf[corpus]

#for line in corpus:
#    print(line)
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf,
                               id2word=dictionary,
                               num_topics=100,
                               update_every=1,
                               eta=0.02,
                               chunksize=10000,
                               passes=10)

print("****TOP TOPICS****")
lda.print_topics(10)

lda.save("tmp/lda_topics.model")
tfidf.save("tmp/tfidf.model")
