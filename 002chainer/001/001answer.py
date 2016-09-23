import numpy as np
from chainer import Variable

x_data = np.array([3, 10, 4], dtype=np.float32)
x = Variable(x_data)
print(x.data)
y = np.linalg.norm(x**2 - 2 * x + 1)
print(y.data)
