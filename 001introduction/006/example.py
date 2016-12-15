from chainer import links as L
from chainer import functions as F
import numpy as np
from chainer import Variable


lin = L.Linear(5, 2)
x = Variable(np.ones((3, 5), dtype=np.float32))
y1 = lin(x)

y2 = F.relu(lin(x))
print(y2.data)
