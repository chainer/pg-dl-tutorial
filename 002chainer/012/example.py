import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Variable
import numpy as np


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


batchsize = 100
train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)

model = MLP(784, 10)
opt = chainer.optimizers.Adam()
opt.setup(model)

train_num = len(train)
for i in range(0, train_num, batchsize):
    batch = train_iter.next()
    x = Variable(np.asarray([s[0] for s in batch]))
    t = Variable(np.asarray([s[1] for s in batch]))
