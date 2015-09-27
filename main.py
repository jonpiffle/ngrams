from language_model import NGramLanguageModel
from probability import LaplaceProbabilityGenerator
from probability import RawProbabilityGenerator


MODELS = [
   {
       'name': 'TriGram LanguageModel with raw probabilities',
       'class': NGramLanguageModel,
       'kwargs': {
           'n': 3,
           'probability_generator': RawProbabilityGenerator,
       },
   },
   {
       'name': 'TriGram LanguageModel with Laplace probabilities',
       'class': NGramLanguageModel,
       'kwargs': {
           'n': 3,
           'probability_generator': LaplaceProbabilityGenerator,
       },
   },
]


def main(model, evaluate=None, unscramble=None):
    print(model)
    print(evaluate)
    print(unscramble)
    language_model = model['class'](**model['kwargs'])
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

    models = dict([(i, m['name']) for i, m in enumerate(MODELS)])
    parser.add_argument(
        '-model',
        metavar='model_number',
        type=int,
        required=True,
        choices=range(len(MODELS)),
        help=str(models),
    )

    evaluate_unscramble = parser.add_mutually_exclusive_group(required=True)
    evaluate_unscramble.add_argument('-evaluate', metavar='file')
    evaluate_unscramble.add_argument('-unscramble', metavar='file')

    args = parser.parse_args().__dict__
    main(MODELS[args.pop('model')], **args)
