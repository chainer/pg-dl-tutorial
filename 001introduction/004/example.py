from chainer import links as L
from chainer import functions as F
import numpy as np
from chainer import Variable


lin = L.Linear(100, 20)
x = Variable(np.ones((10, 100)), dtype=np.float32)
y1 = lin(x)

print(x.data)
print(y1.data)

y2 = F.relu(x)

y3 = F.relu(lin(x))
