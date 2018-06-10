from typing import Iterable, Sequence, TypeVar; T = TypeVar('T')
from itertools import combinations


def iter_pairs(iterable: Iterable[T]) -> Iterable[Sequence[T]]:
    '''Iterate all pairs in *iterable*
    '''
    return combinations(iterable, 2)
