from typing import Iterable, Callable, TypeVar


T = TypeVar('T')
U = TypeVar('U')


def flatmap(f: Callable[[T], Iterable[U]], iterable: Iterable[T])-> Iterable[U]:
    '''
    Apply `f` to each item in `iterable`, and yield from each result.

    >>> list(flatmap(lambda x: [x, x], range(2)))
    [0, 0, 1, 1]

    >>> list(flatmap(range, [2, 3]))
    [0, 1, 0, 1, 2]
    '''
    for x in iterable:
        for item in f(x):
            yield item
