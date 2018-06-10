from typing import Iterable
from collections import defaultdict


class Dictionary:
    '''
    Maintain an id <-> word mapping.
    Before using, it must be `freeze`d.

    >>> dictionary = Dictionary()
    >>> dictionary.add_doc(['dog', 'cat', 'dog', 'fog', 'dog'])
    >>> dictionary.add_doc(['cat', 'cat', 'frog', 'fog'])
    >>> dictionary.add_doc(['fog', 'cat', 'fog', 'fog'])
    >>> dictionary.freeze(min_count=2)
    >>> dictionary.words
    ['cat', 'fog']
    >>> list(dictionary.filtermap(['dog', 'cat', 'fog', 'cat', 'cat']))
    [0, 1, 0, 0]
    '''
    def __init__(self, docs=None):
        self.word_counts = defaultdict(int)
        self._n_docs = 0
        self.freezed = False

        if docs is not None:
            self.add_docs(docs)

    def add_doc(self, doc: Iterable[str]):
        if self.freezed:
            raise RuntimeError('cannot add doc to already freezed Dictionary')
        self._n_docs += 1
        for word in set(doc):
            self.word_counts[word] += 1

    def add_docs(self, docs: Iterable[Iterable[str]]):
        for doc in docs:
            self.add_doc(doc)

    def freeze(self, min_count=0, max_ratio=1):
        self.freezed = True
        self.words = set()

        for word, n in self.word_counts.items():
            if min_count <= n <= self._n_docs * max_ratio:
                self.words.add(word)

        self.words = sorted(self.words)
        self._word2id = {w:i for i, w in enumerate(self.words)}

    def filtermap(self, words: Iterable[str]) -> Iterable[int]:
        if not self.freezed:
            raise RuntimeError('"in" operator can be applied only to freezed Dictionary')
        for word in words:
            if word in self:
                yield self[word]

    def __contains__(self, word):
        if not self.freezed:
            raise RuntimeError('"in" operator can be applied only to freezed Dictionary')

        return word in self._word2id

    def __len__(self):
        if not self.freezed:
            raise RuntimeError('len() operator can be applied only to freezed Dictionary')
        return len(self.words)

    def __getitem__(self, word):
        if not self.freezed:
            raise RuntimeError('"[]" operator can be applied only to freezed Dictionary')

        return self._word2id[word]

    def save(self, filename):
        if not self.freezed:
            raise RuntimeError('can only save freezed Dictionary')

        with open(filename, 'w') as f:
            for word in self.words:
                print(word, file=f)

    @staticmethod
    def load(filename):
        dictionary = Dictionary()
        dictionary.freezed = True
        with open(filename) as f:
            dictionary.words = [l[:-1] for l in f]
        dictionary._word2id = {w:i for i, w in enumerate(dictionary.words)}
        return dictionary
