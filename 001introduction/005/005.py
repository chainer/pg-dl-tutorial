import chainer
from chainer import datasets
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

train, test = chainer.datasets.get_mnist()

batchsize = 100
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)

model = L.Classifier(MLP(784, 10))
opt = chainer.optimizers.Adam()
opt.setup(model)

epoch = 10

# Set up a trainer
updater = training.StandardUpdater(train_iter, opt, device=-1)
trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

# Run the training
trainer.run()
