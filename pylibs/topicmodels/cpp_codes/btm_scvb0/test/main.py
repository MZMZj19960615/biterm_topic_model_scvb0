import sys; sys.path.insert(0, '../bin')
from btm import BTM_SCVB0

if __name__ == '__main__':
    biterms = [
        (0, 1), (2, 3), (0, 1), (2, 5)
    ]
    model = BTM_SCVB0(
        n_topics=10,
        n_word_types=100,
        biterms=biterms,
        seed=123,
    )
    for i in range(100):
        model.update()
    thetak = model.thetak()
    phikv = model.phikv()
    print(thetak)
    print(phikv)
    print(thetak.shape)
    print(phikv.shape)
