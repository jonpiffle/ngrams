import pandas as pd
import numpy as np
import scipy.misc
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
            )]['probabilities']


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

    def __str__(self):
        return "Raw Probability Generator (MLE Counts)"


class LazyProbabilityGenerator(ProbabilityGenerator):
    """
    Probability generator which - for perf/memory reasons, does not compute
    all of its probabilities up front. By definition, these probability models
    expect to not have a probability of 0. If they encounter such a
    probability, then they will default to the result of a lazy_probability
    function.
    """

    def __init__(self, counts):
        super().__init__(counts)

    def get_probabilities(self, state, n=None):
        if n is None:
            n = self.counts.n
        probs = self.probs[n]

        prob = probs[np.logical_and.reduce(
            [probs['word' + str(i+1)] == w for i, w in enumerate(state)],
        )]['probability']

        if not len(prob.values) or prob.values[0] == 0:
            return {0: self.lazy_probability(state, n)}
        else:
            return prob

    def lazy_probability(self, state, n):
        raise NotImplementedError


class LaplaceProbabilityGenerator(LazyProbabilityGenerator):

    def __init__(self, counts, k=1):
        self.k = k
        self.corpus_size = len(counts.get_counts(1))
        self.Ns = {}
        super().__init__(counts)

    def _generate_probabilities(self):
        for i in range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total_possible_ngrams = \
                scipy.misc.comb(self.corpus_size, i) * scipy.misc.factorial(i)
            N = sum(probs['count'].values) + (total_possible_ngrams * self.k)
            self.Ns[i] = N
            probs['probability'] = (probs['count'] + self.k) / N
            probs = probs.drop('count', 1)
            self.probs[i] = probs

    def lazy_probability(self, state, n):
        return self.k / self.Ns[n]

    def __str__(self):
        name = "Laplace Probabilities"
        return "{} with k={}".format(name, self.k)


class AbsoluteDiscountProbabilityGenerator(LazyProbabilityGenerator):

    def __init__(self, counts, D=0):
        self.D = D
        self.corpus_size = len(counts.get_counts(1))
        self.alphas = {}
        super().__init__(counts)

    def _generate_probabilities(self):
        for i in range(1, self.counts.n + 1):
            probs = self.counts.get_counts(i)
            total = sum(probs['count'].values)
            probs['probability'] = (probs['count'] - self.D) / total
            total_possible_ngrams = \
                scipy.misc.comb(self.corpus_size, i) * scipy.misc.factorial(i)
            # if there are k n-grams with counts of zero, then alpha is 1/k.
            # Want to distribute D probability mass across these k unseen
            # elements, so each should get probability (1/k)*D.
            alpha = 1 / (total_possible_ngrams - len(probs))
            self.alphas[i] = alpha
            probs = probs.drop('count', 1)
            self.probs[i] = probs

    def lazy_probability(self, state, n):
        return self.alphas[n] * self.D

    def __str__(self):
        name = "Absolute Discount Probabilities"
        return "{} with D={}".format(name, self.D)
