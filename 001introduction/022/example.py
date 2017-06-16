import chainer
from chainer import datasets
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

import playground


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# create model
model = L.Classifier(MLP(100, 10))

# load dataset
train_full, test_full = chainer.datasets.get_mnist()
train = datasets.SubDataset(train_full, 0, 1000)
test = datasets.SubDataset(test_full, 0, 1000)

# Set up a iterator
batchsize = 100
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)

# Set up an optimizer

# Set up an updater

# Set up a trainer

# Run the trainer

# Check the result
