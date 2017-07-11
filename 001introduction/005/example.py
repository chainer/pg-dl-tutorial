from chainer import functions as F
from chainer import links as L
from chainer import Variable
import numpy as np


lin = L.Linear(5, 2)
x = Variable(np.ones((3, 5), dtype=np.float32))
y1 = lin(x)

print(x.data)
print(y1.data)

y2 = F.relu(x)
y3 = F.relu(lin(x))
