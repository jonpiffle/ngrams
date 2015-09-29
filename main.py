from language_model import NGramLanguageModel
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
         **probability_generator_kwargs):

    language_model = NGramLanguageModel(
        n=n,
        probability_generator=PROBABILITY_GENERATORS[probability_generator],
        **probability_generator_kwargs
    )
    print(language_model)
    if evaluate:
        language_model.evaluate(evaluate)
    elif unscramble:
        with open(unscramble) as f:
            text = f.read()
        language_model.unscramble(text)

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
        'D',
        type=between_zero_and_one,
        help='Amount of probability mass to set aside for unseen words',
    )

    args = parser.parse_args()
    main(**vars(args))
