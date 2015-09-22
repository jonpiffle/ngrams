import os, pickle
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
    def __init__(self, n):
        self.n = n
        self.corpus_builder = CorpusBuilder()
        self.corpus = self.corpus_builder.load_corpus()
        self.counts = defaultdict(lambda: defaultdict(int))
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
        """ Calculates n-gram counts for a specific n """
        counts = defaultdict(int)
        for s in self.corpus:

            # Add special start and end symbols
            s = ["START"] + s + ["END"]
            
            for n_gram in window(s, n):
                counts[tuple(n_gram)] += 1
        return counts

    def load_counts(self, update=False):
        if update or not os.path.exists(self.filename()):
            self.build_counts()
        self.counts = pickle.load(open(self.filename(), 'rb'))

    def filename(self):
        return '%s/%dgram_counts.pickle' % (self.corpus_builder.data_path, self.n)

if __name__ == '__main__':
    NGramCounts(3).load_counts()
