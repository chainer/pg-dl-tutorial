from chainer import optimizers
from chainer import links as L
import numpy as np


x = L.Linear(1, 1, nobias=True)

opt = optimizers.SGD()
opt.setup(x)

coef = np.zeros((1, 1), dtype=np.float32)
print(coef.shape)


def loss(arg):
    return (arg(coef) - 3) ** 4

for i in xrange(100):
    opt.update(loss, x)
    print x.W.data, loss(x).data
