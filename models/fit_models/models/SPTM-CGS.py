from itertools import combinations
import time

import numpy as np

from topicmodels import SPTM_CGS


def iter_epochs(n_word_types, docs, n_topics, seed):
    model = SPTM_CGS(
        n_topics=n_topics,
        n_word_types=n_word_types,
        n_pseudo_docs=1000,
        docs=docs,
        seed=seed
    )

    while True:
        start_time_s = time.time()
        model.update()
        processing_time_s = time.time() - start_time_s

        topic_word_distribution = model.phikv()

        yield dict(topic_word_distribution=topic_word_distribution, processing_time_s=processing_time_s)


if __name__ == '__main__':
    docs = [[0, 2, 3], [32, 444, 3, 32]]
    iter_epochs(n_word_types=1000, docs=docs, n_topics=3, seed=0)
