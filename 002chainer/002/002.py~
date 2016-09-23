import numpy as np
from chainer import Variable

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = x**2 - 2 * x + 1
y.backward()
print(x.grad)
