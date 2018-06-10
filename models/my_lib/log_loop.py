from typing import Iterable, T

import time
import sys


def log_loop(iterable: Iterable[T]) -> Iterable[T]:
    '''
    Return *iterable* unmodified except periodically reporting the number of items processed
    >>> list(log_loop(range(4)))
    [0, 1, 2, 3]
    '''
    p = time.time()
    start_time = p
    n = 0
    for item in iterable:
        n += 1
        t = time.time()
        if t - p > 5:
            p = t
            print('{} items processed (elapsed time: {}s)'.format(n, t - start_time), file=sys.stderr)
        yield item
