from language_model import NGramLanguageModel
from preprocessing import CorpusBuilder
from probability import AbsoluteDiscountProbabilityGenerator
from probability import LaplaceProbabilityGenerator
from probability import RawProbabilityGenerator


PROBABILITY_GENERATORS = {
    'raw': RawProbabilityGenerator,
    'laplace': LaplaceProbabilityGenerator,
    'abs_dis': AbsoluteDiscountProbabilityGenerator,
}


def main(n=1,
         evaluate=None,
         unscramble=None,
         probability_generator=None,
         stemmed=False,
         unstemmed=True,
         **probability_generator_kwargs):

    stemmed = unstemmed or stemmed
    cb = CorpusBuilder(stemmed=stemmed)
    probability_generator_kwargs['corpus_builder'] = cb
    language_model = NGramLanguageModel(
        n=n,
        probability_generator=PROBABILITY_GENERATORS[probability_generator],
        **probability_generator_kwargs
    )
    print(str(language_model))
    if evaluate:
        perplexity = language_model.evaluate(evaluate)
        print("Perplexity: {}".format(perplexity))
    elif unscramble:
        with open(unscramble) as f:
            text = f.read().strip()
        unscrambled = language_model.unscramble(text)
        unscrambled_perplexity = language_model.perplexity(unscrambled)
        scrambled_perplexity = language_model.perplexity(text)
        print("Original Text: {}".format(text))
        print("Unscrambled sentence: {}".format(unscrambled))
        print("Original perplexity: {0}; Unscrambled perplexity: {1}".format(scrambled_perplexity, unscrambled_perplexity))
        print(language_model.text_log_prob(text))
        print(language_model.text_log_prob(unscrambled))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    evaluate_unscramble = parser.add_mutually_exclusive_group(required=True)
    evaluate_help = "File to evaluate model on. To use the model's own test \
        set, use TEST_CORPUS"
    evaluate_unscramble.add_argument(
        '-evaluate',
        metavar='file',
        help=evaluate_help,
    )
    evaluate_unscramble.add_argument('-unscramble', metavar='file')

    stemmed_unstemmed = parser.add_mutually_exclusive_group()
    stemmed_unstemmed.add_argument(
        '-s',
        '--stemmed',
        help='Use stemmed corpus',
        action='store_true',
    )
    stemmed_unstemmed.add_argument(
        '-u',
        '--unstemmed',
        help='Use unstemmed corpus',
        default=False,
        action='store_false',
    )

    def nonnegative_int(string):
        try:
            val = int(string)
            if val > 0:
                return val
        except ValueError:
            pass
        message = "{} must be a nonnegative integer.".format(string)
        raise argparse.ArgumentTypeError(message)

    def between_zero_and_one(string):
        try:
            val = float(string)
            if 0 <= val <= 1:
                return val
        except ValueError:
            pass
        message = "{} must be between zero and one.".format(string)
        raise argparse.ArgumentTypeError(message)

    parser.add_argument(
        '-n',
        '--n',
        help='Which n-gram model to use',
        type=nonnegative_int,
        default=1,
    )

    subparsers = parser.add_subparsers(
        dest='probability_generator',
        help='options for language models',
    )

    raw_parser = subparsers.add_parser('raw', help='Raw Probability Model')

    laplace_parser = subparsers.add_parser(
        'laplace',
        help='Laplace Probability Model',
    )
    laplace_parser.add_argument(
        '-k',
        '--k',
        type=nonnegative_int,
        default=1,
        help='Amount to adjust counts by',
    )

    absolute_discount_parser = subparsers.add_parser(
        'abs_dis',
        help='Absolute Discount Probability Model',
    )
    absolute_discount_parser.add_argument(
        '-D',
        '--D',
        type=between_zero_and_one,
        help='Amount of probability mass to set aside for unseen words',
        required=False,
        default=0.3,
    )

    args = parser.parse_args()
    main(**vars(args))
