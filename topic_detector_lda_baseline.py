"""
    Detect topic of new document for baseline LDA model.
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
import re
import data_prep as dp
import sys

if __name__ == "__main__":
    total = len(sys.argv)
    if total < 2:
        print("usage: python topic_detector_lda_baseline.py  input.tsv")
        exit(1)
    # Get the arguments list
    input_data = sys.argv[1]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)
    pattern = re.compile('[&!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]')
    dictionary = corpora.Dictionary.load_from_text('tmp/dialogs4.dict')
    corpus = corpora.MmCorpus('tmp/dialogs4-corpus.mm')

    tfidf = models.TfidfModel.load("tmp/tfidf.model")
    lda = models.ldamodel.LdaModel.load("tmp/lda_topics.model")

    new_doc = dp.read_file(input_data)

    new_vec = dictionary.doc2bow(pattern.sub(" ", new_doc.lower()).split())
    new_vec = tfidf[new_vec]
    topics = lda.get_document_topics(new_vec)
    topics = reversed(sorted(topics, key=lambda x: x[1]))
    print("TOPICS:")
    for t in topics:
        terms = lda.get_topic_terms(t[0], 4)
        unpack, _ = zip(*terms)
        words = [dictionary[x] for x in unpack]
        s = "-".join(words)
        print("topic ", t[0], " ", s, " probablity:", t[1])
