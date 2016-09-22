from chainer import optimizers
from chainer import links as L
from chainer import Variable
import numpy as np


class Linear(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 1),
        )

    def __call__(self, x):
        return self.l1(x)


def f(x):
    return 5.*x + 10

x = np.linspace(-10, 10, num=1001)
y = f(x) + 5.*np.random.randn()

model = Linear()

opt = optimizers.SGD()
opt.Setup(model)
for epoch in xrange(100):
    perm = np.random.perm(len(x))
    for i in xrange(len(x)):
        x_i = Variable(x[perm[i]])
        y_i = Variable(y[perm[i]])
        ## Write Here