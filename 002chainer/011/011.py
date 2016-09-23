from chainer import datasets

train, test = datasets.get_mnist()

for i in xrange(0, train_num, batchsize):
  batch = train_iter.next()