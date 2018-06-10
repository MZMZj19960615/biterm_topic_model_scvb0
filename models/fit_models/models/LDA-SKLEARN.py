from itertools import combinations
import time
from collections import Counter

import numpy as np
from random import Random
from toolz.functoolz import compose

from my_lib import flatmap
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import lil_matrix


def iter_epochs(n_word_types, docs, n_topics, seed):
    D = len(docs)
    V = n_word_types
    docs = [list(Counter(doc).items()) for doc in docs]
    X = lil_matrix((D, V), dtype=np.int)
    for d, doc in enumerate(docs):
        for v, c in doc:
            X[d, v] = c
    X = X.tocsr()
    model = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', random_state=seed)
    while True:
        start_time_s = time.time()
        model.partial_fit(X)
        processing_time_s = time.time() - start_time_s
        phikv = model.components_
        yield dict(topic_word_distribution=phikv, processing_time_s=processing_time_s)


if __name__ == '__main__':
    docs = [[0, 2, 3], [8, 8, 3, 2]]
    for info in iter_epochs(n_word_types=10, docs=docs, n_topics=3, seed=0):
        print(info)
