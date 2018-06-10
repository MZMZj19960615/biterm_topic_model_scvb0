from itertools import combinations
import time

import numpy as np
from toolz.functoolz import compose

from topicmodels import BTM_SCVB0
from my_lib import iter_pairs
from my_lib import flatmap
from my_lib import biterms2topic_vector



def iter_epochs(n_word_types, docs, n_topics, seed):
    D = len(docs)
    K = n_topics
    def iter_doc_biterms():
        return map(compose(list, iter_pairs, sorted, set), docs)

    biterms = list(flatmap(lambda x: x, iter_doc_biterms()))

    model = BTM_SCVB0(
        n_topics=n_topics,
        n_word_types=n_word_types,
        biterms=biterms,
        alpha=0.1,
        beta=0.01,
        tau=10,
        seed=seed
    )

    while True:
        start_time_s = time.time()
        model.update()
        processing_time_s = time.time() - start_time_s

        phikv = model.phikv()

        yield dict(topic_word_distribution=phikv, processing_time_s=processing_time_s)


if __name__ == '__main__':
    docs = [[0, 2, 3], [32, 444, 3, 32]]
    iter_epochs(n_word_types=1000, docs=docs, n_topics=3, seed=0)
