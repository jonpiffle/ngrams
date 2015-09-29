import os, pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from preprocessing import CorpusBuilder
from utils import window, START_SYMBOL, END_SYMBOL

class NGramCounts(object):
    def __init__(self, n, corpus_builder=None):
        self.n = n

        # Default CorpusBuilder
        if corpus_builder is None:
            corpus_builder = CorpusBuilder()

        self.corpus_builder = corpus_builder
        self.counts = {}

        # set counts data
        self.load_counts()

    def get_counts(self, n=None):
        if n is None:
            n = self.n

        return self.counts[n]

    def build_counts(self):
        """
        Calculates n-gram counts for all n-grams <= n.
        The lower-order counts will be necessary for some models.
        Pickles and stores resulting dict
        """
        for i in range(1, self.n + 1):
            self.counts[i] = self._build_counts(i)
        pickle.dump(dict(self.counts), open(self.filename(), 'wb'))

    def _build_counts(self, n):
        """
        Calculates n-gram counts for a specific n
        Stores them in a pandas DataFrame as:
            | word1 | word2 | ... | wordn | count |
        """
        # Generate a dictionary of {n-gram tuple: count}
        counts = defaultdict(int)

        # Reload corpus every time so that it doesn't need to permanently stay in memory
        train_corpus, test_corpus = self.corpus_builder.load_corpus()
        for s in train_corpus:

            # Add special start and end symbols
            s = [START_SYMBOL] + s + [END_SYMBOL]

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

    def lexicon_to_csv(self, output_file):
        unigram_counts = self.get_counts(n=1)
        with open(output_file, 'w') as f:
            f.write(unigram_counts.sort('count', ascending=False).to_csv(index=False))

if __name__ == '__main__':
    NGramCounts(1, corpus_builder=CorpusBuilder(stemmed=True)).write_lexicon_to_file('stemmed_lexicon.csv')
    NGramCounts(1, corpus_builder=CorpusBuilder(stemmed=False)).write_lexicon_to_file('unstemmed_lexicon.csv')
