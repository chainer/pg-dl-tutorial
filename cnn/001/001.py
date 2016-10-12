from chainer import datasets

train, test = datasets.get_cifar100()

x, y = train[0]
print x.shape