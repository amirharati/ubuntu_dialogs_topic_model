"""
    Detect topic of new document for doc2vec
"""
import UbuntuCorpus as UC
from gensim import corpora, models, similarities
import logging
import re
import data_prep as dp
import sys
import pickle

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)
    total = len(sys.argv)
    if total < 2:
        print("usage: python topic_detector_doc2-vec.py  input.tsv")
        exit(1)
    # Get the arguments list
    input_data = sys.argv[1]

    fo = open("tmp/vec_cluster.pk", "rb")
    VC = pickle.load(fo)
    fo.close()
    model = models.doc2vec.Doc2Vec.load("tmp/doc2vec.model")

    pattern = re.compile('[&!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]')
    dictionary = corpora.Dictionary.load_from_text('tmp/dialogs4.dict')
    new_doc = dp.read_file(input_data)
    v = []
    words = pattern.sub(" ", new_doc.lower()).split()
    for word in words:
        if word in dictionary.token2id:
            v.append(word)
    inferred_docvec = model.infer_vector(v)
    topics = VC.predict(inferred_docvec)

    print("TOPICS:")
    for t in topics:
        print("topic ", t[0], " ", t[1], " probablity:", t[2])

