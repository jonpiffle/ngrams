import codecs
import csv
import glob
import os
import pickle

class CorpusBuilder(object):
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.text_dir = self.data_path + '/wordLemPoS'

    def _build_corpus(self, stemmed=False):
        """
        Reads in a text corpus stored as a tab separated file of: word, lemma, pos
        Filters NUL bytes and unwanted symbols
        Splits corpus into sentences based on punctuation ('.', '!', '?')
        Pickles full list of sentences
        """

        textfiles = glob.glob(self.text_dir + '/*.txt')
        sentences = []
        for filename in textfiles:
            with open(filename, 'r') as f:
                r = csv.DictReader((x.replace('\0', '') for x in f), dialect='excel-tab', quoting=csv.QUOTE_NONE, fieldnames=['word', 'lemma', 'pos'])

                sentence = []
                for row in r:
                    if '#' in row['word'] or '@' in row['word']:
                        continue
                    elif '.' in row['pos'] or '!' in row['pos'] or '?' in row['pos']:
                        sentences.append(sentence)
                        sentence = []
                    elif stemmed:
                        if row['lemma'] != '':
                            sentence.append(row['lemma'])
                    else:
                        sentence.append(row['word'])

        pickle.dump(sentences, open(self.filename(stemmed), 'wb'))

    def load_corpus(self, stemmed=False, update=False):
        """ Returns corpus as a list of sentences. Loads from pickle if available. """
        if update or not os.path.exists(self.filename(stemmed)):
            self._build_corpus(stemmed)
        return pickle.load(open(self.filename(stemmed), 'rb'))

    def filename(self, stemmed):
        """ Returns filename of the pickled corpus """
        suffix = 'stemmed' if stemmed else 'unstemmed'
        filename = '%s/corpus_%s.pickle' % (self.data_path, suffix)
        return filename

if __name__ == '__main__':
    corpus = CorpusBuilder().load_corpus()
    print(corpus)
