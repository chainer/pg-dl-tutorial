import chainer
from chainer import functions as F
from chainer import links as L

# make your network


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        pass

    def __call__(self, x):
        pass


model = L.Classifier(MLP(784, 2))

# print out namedlinks
for l in model.namedlinks():
    print(l)
