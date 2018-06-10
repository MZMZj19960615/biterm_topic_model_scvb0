import numpy as np


def biterms2topic_vector(biterms, thetak, phikv):
    '''
    Calculate pseudo topic posterior for each document according to

    Yan, Xiaohui, et al. "A biterm topic model for short texts."
    Proceedings of the 22nd international conference on World Wide Web. ACM, 2013.

    `biterms`: biterms contained in a document
    `thetak`: global topic distribution
    `phikv`: topic-word distributions

    '''
    K, = thetak.shape
    def biterm_posterior(biterms):
        dist = thetak[:] * phikv[:, biterm[0]] * phikv[:, biterm[1]]
        dist /= dist.sum()
        return dist
    topic_posterior = np.zeros(K)
    for biterm in biterms:
        topic_posterior += biterm_posterior(biterm)

    s = topic_posterior.sum()
    if s != 0:
        topic_posterior /= s
    else:
        topic_posterior[:] = 1 / K
    return topic_posterior
