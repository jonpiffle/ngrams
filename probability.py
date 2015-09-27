import pandas as pd
import numpy as np
from ngram import NGramCounts
from utils import window


class ProbabilityGenerator(object):

    def __init__(self, counts):
        self.counts = counts
        self.probs = {}
        self._generate_probabilities()

    def _generate_probabilities(self):
        raise NotImplementedError

    def get_probability(self, state, action):
        raise NotImplementedError

    def get_probabilities(self, state, n=None):
        """
        For a partial state, returns all probabilities that could finish state
        e.g. state = ('at', '4:23') with counts of 3-grams
        returns the only 3-gram matching those two elements in the first 2
        positions:
            word1 word2 word3  probability
            at    4:23  pm     0.000003
        """
        if n is None:
            n = self.counts.n
        probs = self.probs[n]

        if len(state) == 1:
            return probs[probs['word1'] == state[0]]
        else:
            return probs[np.logical_and.reduce(
                [probs['word' + str(i+1)] == w for i, w in enumerate(state)],
            )]


class RawProbabilityGenerator(ProbabilityGenerator):

    def __init__(self, counts):
        super().__init__(counts)

    def _generate_probabilities(self):
        for i in range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total = sum(probs['count'].values)
            probs['probability'] = probs['count'] / total
            probs = probs.drop('count', 1)
            self.probs[i] = probs


class LaplaceProbabilityGenerator(ProbabilityGenerator):

    def __init__(self, counts, k=1):
        self.k = k
        super().__init__(counts)

    def _generate_probabilities(self):
        for i in range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total = sum(probs['count'].values) + (len(probs) * self.k)
            probs['probability'] = (probs['count'] + self.k) / total
            probs = probs.drop('count', 1)
            self.probs[i] = probs


class AbsoluteDiscountProbabilityGenerator(ProbabilityGenerator):

    def __init__(self, counts, alpha=1, D=None):
        self.alpha = alpha
        self.D = D
        super().__init__()

    def _generate_probablities(self):
        for i range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total = sum(probs['count'].values)
            if total > 0:
                probs['probability'] = (probs['count'] - self.D) / total
            else:
                probs['probability'] = self.alpha * self.D
            probs = probs.drop('count', 1)
            self.probs[i] = probs
