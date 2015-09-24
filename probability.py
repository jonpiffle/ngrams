import pandas as pd
import numpy as np
from ngram import NGramCounts
from utils import window

class ProbabilityGenerator(object):
    def __init__(self, counts):
        self.counts = counts
        self.probs = None

    def get_probability(self, state, action):
        raise NotImplementedError

    def get_probabilities(self, state):
        raise NotImplementedError

class UnsmoothedProbabilityGenerator(ProbabilityGenerator):
    def __init__(self, counts):
        self.counts = counts
        self.probs = {}

        # set self.probs
        self._generate_probabilities()

    def get_probabilities(self, state, n=None):
        """
        For a partial state, returns all probabilities that could finish state
        e.g. state = ('at', '4:23') with counts of 3-grams
        returns the only 3-gram matching those two elements in the first 2 positions:
            word1 word2 word3  probability
            at    4:23  pm     0.000003
        """
        if n is None:
            n = self.counts.n
        probs = self.probs[n]

        if len(state) == 1:
            return probs[probs['word1'] == state[0]]
        else:
            return probs[np.logical_and.reduce([probs['word' + str(i+1)] == w for i, w in enumerate(state)])]

    def _generate_probabilities(self):
        for i in range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total = sum(probs['count'].values)
            probs['probability'] = probs['count'] / total
            probs = probs.drop('count', 1)
            self.probs[i] = probs
