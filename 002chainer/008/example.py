from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import Variable
import numpy as np


class Linear(Chain):
    def __init__(self):
        super(Linear, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(1, 1)

    def __call__(self, x):
        return self.l1(x)


def f(x):
    return 5. * x + 10


x = np.linspace(-10, 10, num=1001)
y = f(x) + 5. * np.random.randn(1001)

model = Linear()

opt = optimizers.SGD()
opt.setup(model)
for epoch in range(100):
    perm = np.random.permutation(len(x))
    for i in range(len(x)):
        x_i = Variable(np.array([[x[perm[i]]]], 'f'))
        y_i = Variable(np.array([[y[perm[i]]]], 'f'))
        # Write Here
