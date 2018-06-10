from itertools import combinations
import time
from collections import Counter

import numpy as np
from random import Random
from toolz.functoolz import compose

from my_lib import flatmap
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_topic_word_distribution(model, n_topics, n_word_types):
    phikv = np.zeros((n_topics, n_word_types))
    for k, topic in model.show_topics(num_topics=n_topics, num_words=n_word_types, formatted=False):
        for word_id, p in topic:
            phikv[k, word_id] = p
    return phikv

def iter_epochs(n_word_types, docs, n_topics, seed):
    docs = [list(Counter(doc).items()) for doc in docs]
    D = len(docs)
    K = n_topics
    V = n_word_types
    model = gensim.models.LdaModel(corpus=None, num_topics=K, id2word={i:i for i in range(V)}, alpha='symmetric', minimum_probability=0, random_state=seed)
    prng = Random(seed)

    while True:
        prng.shuffle(docs)
        start_time_s = time.time()
        model.update(docs)
        processing_time_s = time.time() - start_time_s

        phikv = get_topic_word_distribution(model, n_topics=n_topics, n_word_types=n_word_types)

        yield dict(topic_word_distribution=phikv, processing_time_s=processing_time_s)


if __name__ == '__main__':
    docs = [[0, 2, 3], [8, 8, 3, 2]]
    for info in iter_epochs(n_word_types=10, docs=docs, n_topics=3, seed=0):
        print(info)
