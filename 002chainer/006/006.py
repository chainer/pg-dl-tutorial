from chainer import Chain, ChainList
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


class MyChainList(ChainList):
    def __init__(self):
        super(MyChain, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)

c = MyChain()
c2 = MyChainList
