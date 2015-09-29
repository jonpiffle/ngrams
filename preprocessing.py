import codecs
import csv
import glob
import os
import pickle
from sklearn.cross_validation import train_test_split

class CorpusBuilder(object):
    def __init__(self, data_path='data', stemmed=False):
        self.data_path = data_path
        self.text_dir = self.data_path + '/wordLemPoS'
        self.stemmed = stemmed
        self.stem_map = None

    def _build_corpus(self):
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
                    elif self.stemmed:
                        if row['lemma'] != '':
                            sentence.append(row['lemma'])
                    else:
                        sentence.append(row['word'])

        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)
        pickle.dump(train_sentences, open(self.filename(), 'wb'))
        pickle.dump(test_sentences, open(self.test_filename(), 'wb'))

    def stem(self, word):
        """ Returns the stem of the word as defined by the corpus, or the word if not in the stem_map """
        if self.stem_map is None:
            self.stem_map = self.load_stem_map()
        return self.stem_map[word] if word in self.stem_map else word

    def _build_stem_map(self):
        """ Builds and pickles a dictionary of {word: stem} using the corpus """
        stem_map = {}
        textfiles = glob.glob(self.text_dir + '/*.txt')
        for filename in textfiles:
            with open(filename, 'r') as f:
                r = csv.DictReader((x.replace('\0', '') for x in f), dialect='excel-tab', quoting=csv.QUOTE_NONE, fieldnames=['word', 'lemma', 'pos'])

                for row in r:
                    if '#' in row['word'] or '@' in row['word']:
                        continue
                    else:
                        stem_map[row['word']] = row['lemma']
        pickle.dump(stem_map, open(self.stem_map_filename(), 'wb'))

    def load_stem_map(self, update=False):
        """ Returns the mapping of {word: stem} built from the corpus. Loads from pickle if available """
        if update or not (os.path.exists(self.stem_map_filename())):
            self._build_stem_map()
        return pickle.load(open(self.stem_map_filename(), 'rb'))

    def load_corpus(self, update=False):
        """ Returns train, test corpus as a list of sentences. Loads from pickle if available. """
        if update or not (os.path.exists(self.filename()) and os.path.exists(self.test_filename())):
            self._build_corpus()
        return pickle.load(open(self.filename(), 'rb')), pickle.load(open(self.test_filename(), 'rb'))

    def test_filename(self):
        """ Returns filename of the pickled test corpus """
        return self.filename().split('.')[0] + '_test.pickle'

    def filename(self):
        """ Returns filename of the pickled corpus """
        suffix = 'stemmed' if self.stemmed else 'unstemmed'
        filename = '%s/corpus_%s.pickle' % (self.data_path, suffix)
        return filename

    def stem_map_filename(self):
        """ Returns filename of the pickled dictionary {word: stem} """
        return self.data_path + '/stem_map.pickle'

if __name__ == '__main__':
    cb = CorpusBuilder()
    train, test = cb.load_corpus()
    stem_map = cb.load_stem_map()
    print(len(train))
    print(len(test))
    print(cb.stem('factions'))

