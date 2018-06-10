import sys; sys.path.insert(0, '../bin')
from mod import BTM_CGS


model = BTM_CGS(n_topics=10, n_word_types=10, biterms=[(0, 1), (2, 5), (0, 5)])

for iteration in range(1000):
    print('iteration: {}'.format(iteration + 1))
    model.update()

print(model.thetak())
print(model.phikv())
