"""
    Display top 10 topics.
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
import re
import data_prep as dp

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)

    dictionary = corpora.Dictionary.load_from_text('tmp/dialogs4.dict')
    corpus = corpora.MmCorpus('tmp/dialogs4-corpus.mm')
    lda = models.ldamodel.LdaModel.load("tmp/lda_topics.model")

    top_topics = lda.print_topics(10, 4)

    print("**TOP TOPICS**")
    for t in top_topics:
        terms = lda.get_topic_terms(t[0], 4)
        unpack, _ = zip(*terms)
        words = [dictionary[x] for x in unpack]
        s = "-".join(words)
        print("topic ", t[0], " ", s,)
