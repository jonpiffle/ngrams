import numpy as np

from ngram import NGramCounts
from probability import LaplaceProbabilityGenerator
from probability import RawProbabilityGenerator
from utils import window, START_SYMBOL, END_SYMBOL


def permutations(iterable):
    if len(iterable) == 0:
        yield []
    else:
        for i, e in enumerate(iterable):
            others = np.concatenate(
                [np.arange(0, i), np.arange(i + 1, len(iterable))],
            )
            l = [e]
            for p in permutations(iterable[others]):
                l[1:] = list(p)
                yield l


class LanguageModel(object):

    def __init__(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def unscramble(self, text):
        self.cache = {}  # cache may get quite large if not cleared
        words = text.split()
        best_sentence, best_prob = None, 0
        for p in permutations(np.array(words)):
            sentence = ' '.join(p)
            prob = self.text_probability(sentence)
            if prob > best_prob:
                best_sentence, best_prob = sentence, prob
        return best_sentence

    def text_probability(self, text):
        raise NotImplementedError


class NGramLanguageModel(LanguageModel):

    def __init__(self,
                 n=3,
                 probability_generator=RawProbabilityGenerator,
                 **kwargs):
        self.n = n
        self.ngram_counts = NGramCounts(self.n)
        self.probability_generator = probability_generator(
            self.ngram_counts,
            **kwargs
        )
        self.cache = {}

    def evaluate(self):
        pass

    def text_probability(self, text):
        """
        Returns the probability of the text as generated by:
            Prod( Pr(w_k | w_k - 1, ..., w_k - (n - 1)) )
        """

        text = [START_SYMBOL] + text.split() + [END_SYMBOL]
        running_prob = 1
        for w in window(text, self.n, left_nulls=True):
            # for first N - 1 words, have to use a lower order model
            n = self.n - len([a for a in w if a is None])
            w = [a for a in w if a is not None]
            cache_key = tuple(w + [n])
            if cache_key in self.cache:
                probability = self.cache[cache_key]
            else:
                try:
                    probability = self.probability_generator.get_probabilities(
                        w,
                        n=n,
                    )['probability'].values[0]
                except:
                    probability = 0
                self.cache[cache_key] = probability

            print(w, n, probability)
            running_prob *= probability
            if running_prob == 0:
                break
        print(text, running_prob)
        return running_prob


if __name__ == '__main__':
    ng = NGramLanguageModel(probability_generator=LaplaceProbabilityGenerator)
    # print(ng.probability_generator.probs.keys())
    # ng.text_probability('What you only need to ask')
    print(ng.unscramble('you only need What to ask'))
