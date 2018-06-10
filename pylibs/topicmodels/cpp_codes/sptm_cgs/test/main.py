import json
import numpy as np
import os

from mod import SPTM

import sys

seed = 0
try:
    seed = int(sys.argv[1])
except:
    pass

n_topics = 10
n_word_types = 1000
n_pseudo_docs = 10
docs = [
    [3, 0, 1, 3],
    [0, 3],
    [0, 3, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4],
]

model = SPTM(
    n_topics=n_topics,
    n_word_types=n_word_types,
    n_pseudo_docs=n_pseudo_docs,
    docs=docs,
    seed=seed
)

model.update()
print(model.log_marginalized_joint())
model.update()
print(model.log_marginalized_joint())

print(model.phikv())
print(model.ld())
print(model.thetak())
