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
        self.probs = self._generate_probabilities()

    def get_probabilities(self, state):
        """
        For a partial state, returns all probabilities that could finish state
        e.g. state = ('at', '4:23') with counts of 3-grams
        returns the only 3-gram matching those two elements in the first 2 positions:
            word1 word2 word3  probability
            at    4:23  pm     0.000003
        """
        return self.probs[np.logical_and(*[self.probs['word' + str(i+1)] == w for i, w in enumerate(state)])]

    def _generate_probabilities(self):
        N = len(self.counts.get_counts().index)
        probs = self.counts.get_counts()
        probs['probability'] = probs['count'] / N
        probs = probs.drop('count', 1)
        return probs
