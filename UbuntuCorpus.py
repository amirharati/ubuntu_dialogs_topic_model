"""
    Define Ubuntu Corpus class.
    Provide iterator to read the corpus.
    Also perform data cleanup.
"""
import logging
import nltk
from nltk.corpus import stopwords
from gensim import corpora
import re
from six import iteritems
from gensim import corpora, models


class UbuntuCorpus:
    def __init__(self, corpus_text, dictout, tagged_document=False):
        """define a serilizd corpus.
           we represet each documnet with a bag of words representation.
        """
        # download stopwords
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s'
                            , level=logging.INFO)
        nltk.download("stopwords")
        self.tagged_document = tagged_document
        # remove these chars from data
        self.pattern = re.compile('[&!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]')

        self.corpus_text = corpus_text
        _len = 0
        for line in open(self.corpus_text):
            _len += 1
        self.length = _len
        stopwords_list = stopwords.words("english")
        # application specific stop words.
        custom_stops = ['hey', 'hello', 'hi', 'thanks', 'thank', 'heh', 'need', 'want', 'help', 'told', 'said', 'good', 'ok', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within', 'today', 'please', 'pls', 'use', 'morning', 'anyone', 'question', 'ask', 'dude', 'knew', 'anymore', 'hehe', 'ah', 'total', 'ops', 'oops', 'know', 'love', 'huh', 'month', 'ignore', 'ahh', 'funny', 'yo', 'yeah', 'yea', 'yes', 'uh', 'sorry', 'sry', 'alternate']
        self.dictionary = corpora.Dictionary(self.pattern.sub(" ", line.lower()).split() for line in open(self.corpus_text))
        stop_ids = [self.dictionary.token2id[stopword]
                    for stopword in stopwords_list
                    if stopword in self.dictionary.token2id]
        custom_ids = [self.dictionary.token2id[stopword]
                    for stopword in custom_stops
                    if stopword in self.dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in iteritems(self.dictionary.dfs) if docfreq == 1]
        freq_ids = [tokenid for tokenid, docfreq in iteritems(self.dictionary.dfs) if docfreq > self.length / 5]
        self.dictionary.filter_tokens(stop_ids + once_ids + custom_ids + freq_ids)
        # re-distibute the ids
        self.dictionary.compactify()
        self.dictionary.save_as_text(dictout)

    def __iter__(self):
        self.index = -1
        for line in open(self.corpus_text):
            self.index += 1
            if self.tagged_document is False:
                v = self.dictionary.doc2bow(self.pattern.sub(" ", line.lower()).split())
                if (v == []):
                    print('document is empty and will be removed!')
                    self.length -= 1
                else:
                    yield v
            else:
                v = []
                words = self.pattern.sub(" ", line.lower()).split()
                for word in words:
                    if word in self.dictionary.token2id:
                        v.append(word)
                if (v == []):
                    #print('document is empty and will be removed!')
                    self.index -= 1
                    self.length += 1
                else:
                    yield models.doc2vec.TaggedDocument(v, [self.index])

    def __len__(self):
        return self.length

