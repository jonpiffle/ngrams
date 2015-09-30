import math
import numpy as np
from itertools import permutations

from preprocessing import CorpusBuilder
from ngram import NGramCounts
from probability import LaplaceProbabilityGenerator
from probability import RawProbabilityGenerator
from utils import window, START_SYMBOL, END_SYMBOL


class LanguageModel(object):

    def __init__(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def unscramble(self, text):
        self.cache = {}  # cache may get quite large if not cleared
        if self.ngram_counts.corpus_builder.stemmed:
            unstemmed_words = np.array(text.split())
            words = np.array(self.ngram_counts.corpus_builder.stem(text)[0].split())
        else:
            words = np.array(text.split())
            unstemmed_words = words

        best_sentence_indices, best_log_prob = [], float('-inf')
        for p in permutations(range(len(words))):
            p = list(p)
            sentence = ' '.join(words[p])
            prob = self.text_log_prob(sentence)
            if prob > best_log_prob:
                best_sentence_indices, best_log_prob = p, prob

        return ' '.join(list(unstemmed_words[best_sentence_indices]))

    def text_log_prob(self, text):
        raise NotImplementedError


class NGramLanguageModel(LanguageModel):

    def __init__(self,
                 n=3,
                 probability_generator=RawProbabilityGenerator,
                 **kwargs):
        self.n = n
        corpus_builder = kwargs.pop('corpus_builder', None)
        self.ngram_counts = NGramCounts(self.n, corpus_builder=corpus_builder)
        self.probability_generator = probability_generator(
            self.ngram_counts,
            **kwargs
        )
        self.cache = {}

    def __str__(self):
        gram = "{}-gram".format(self.n)
        stemmed = "Stemmed" if self.ngram_counts.corpus_builder.stemmed else "Unstemmed"
        prob = str(self.probability_generator)
        return "Model: {}; {}; {}".format(gram, stemmed, prob)

    def evaluate(self, test_text_file=None):
        """ Returns the perplexity of the test text (if given) otherwise the corpus test set """
        test_text = self._load_test_text(test_text_file)
        return self.perplexity(test_text)

    def _load_test_text(self, test_text_file):
        """
        Takes the filename of a text corpus to be evaluated and returns the loaded text as sentences
        If the filename is None, returns the test set from the train/test corpus split
        """

        if test_text_file == 'TEST_CORPUS':
            corpus_builder = self.ngram_counts.corpus_builder 
            train_corpus, test_corpus = corpus_builder.load_corpus()
            sentences = [" ".join(s) for s in test_corpus]
            return sentences
        else:
            with open(test_text_file, 'r') as f:
                text = f.read().strip()
            sentences = text.split('.')

            # If using a stemmed model, need to stem test input
            if self.ngram_counts.corpus_builder.stemmed:
                sentences = self.ngram_counts.corpus_builder.stem(sentences)

            return sentences

    def perplexity(self, text):
        """
        Computes the perplexity of a piece of text as:
            (prod(1/Pr(w|words before))^(1/N)
        Takes either a string or a list of strings (sentences)
        """

        if isinstance(text, str):
            text = [text]

        running_log_prob = 0

        # Text length = number of words + start and end symbol for each sentence
        text_len = sum([len(s.split()) + 2 for s in text])
        print(text_len)

        for i, sentence in enumerate(text):
            sentence_log_prob = self.text_log_prob(sentence)
            running_log_prob += sentence_log_prob
            print(i, running_log_prob)

        log_perplexity = - 1 / text_len * running_log_prob
        perplexity = math.exp(log_perplexity)
        return perplexity

    def text_log_prob(self, text):
        """
        Returns the probability of the text as generated by:
            Prod( Pr(w_k | w_k - 1, ..., w_k - (n - 1)) )
        """

        text = [START_SYMBOL] + text.split() + [END_SYMBOL]
        running_prob = 0
        for w in window(text, self.n, left_nulls=True):
            # for first N - 1 words, have to use a lower order model
            n = self.n - len([a for a in w if a is None])
            w = [a for a in w if a is not None]
            cache_key = tuple(w + [n])
            if cache_key in self.cache:
                probability = self.cache[cache_key]
            else:
                probability = self.probability_generator.get_probabilities(
                    w,
                    n=n,
                )
                index = [k for k in probability.keys()][0]
                probability = probability[index]
                self.cache[cache_key] = probability

            if probability == 0:
                return float('-inf')

            running_prob += math.log(probability)
            print(w, n, probability, running_prob)
        print(text, running_prob)
        return running_prob

if __name__ == '__main__':
    cb = CorpusBuilder(stemmed=True)
    ng = NGramLanguageModel(probability_generator=LaplaceProbabilityGenerator, corpus_builder=cb)
    # print(ng.probability_generator.probs.keys())
    # ng.text_log_prob('What you only need to ask')
    print(ng.unscramble('needed You only'))
    # print(ng.perplexity('you only need What to ask'))
    # print(ng.perplexity('What you only need to ask'))
    # print(ng.evaluate())
