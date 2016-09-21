from chainer import Chain
from chainer import links as L

class MyChain(Chain):
  def __init__(self):
    super(MyChain, self).__init__(
      l1=L.Linear(4, 3),
      l2=L.Linear(3, 2),
    )

  def __call__(self, x):
    h = self.l1(x)
    return self.l2(h)

c = MyChain()

print(c.namedparams())