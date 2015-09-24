from itertools import tee

START_SYMBOL = '__START__'
END_SYMBOL = '__END__'

def window(iterable, size, left_nulls=False):
    """
    Iterates over an iterable size elements at a time
    [1, 2, 3, 4, 5], 3 ->
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    """

    # Pad left with None's so that the the first iteration is [None, ..., None, iterable[0]]
    if left_nulls:
        iterable = [None] * (size - 1) + iterable

    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)
