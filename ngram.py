import os, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from preprocessing import CorpusBuilder
from itertools import tee

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

class NGramCounts(object):
    def __init__(self, n, corpus_builder=None):
        self.n = n

        # Default CorpusBuilder
        if corpus_builder is None:
            corpus_builder = CorpusBuilder()

        self.corpus_builder = corpus_builder
        self.corpus = self.corpus_builder.load_corpus()
        self.counts = {}
        # set count data
        self.load_counts()

    def build_counts(self):
        """
        Calculates n-gram counts for all n-grams <= n.
        The lower-order counts will be necessary for some models.
        Pickles and stores resulting dict
        """
        for i in range(1, self.n + 1):
            self.counts[i] = self._build_counts(i)
        pickle.dump(dict(self.counts)   , open(self.filename(), 'wb'))

    def _build_counts(self, n):
        """
        Calculates n-gram counts for a specific n
        Stores them in a pandas DataFrame as:
            | word1 | word2 | ... | wordn | count |
        """
        # Generate a dictionary of {n-gram tuple: count}
        counts = defaultdict(int)
        for s in self.corpus:

            # Add special start and end symbols
            s = ["START"] + s + ["END"]

            for n_gram in window(s, n):
                counts[tuple(n_gram)] += 1

        # Converts to a list of dictionaries [{'word1': <word1>, ..., 'wordn': <wordn>, 'count': <count>}]
        lst_of_dicts = []
        for k, v in counts.items():
            d = {'word' + str(i+1): w for i, w in enumerate(k)}
            d['count'] = v
            lst_of_dicts.append(d)

        # Converts list of dicts to a dataframe
        return pd.DataFrame(lst_of_dicts)

    def load_counts(self, update=False):
        if update or not os.path.exists(self.filename()):
            self.build_counts()
        self.counts = pickle.load(open(self.filename(), 'rb'))

    def filename(self):
        suffix = 'stemmed' if self.corpus_builder.stemmed else 'unstemmed'
        return '%s/%dgram_counts_%s.pickle' % (self.corpus_builder.data_path, self.n, suffix)

if __name__ == '__main__':
    n = NGramCounts(3)
    print(n.counts[3])
