import numpy as np
from chainer import Variable

x_data = np.array([[3, 10, 4], [-5, 20, 4]], dtype=np.float32)
x = Variable(x_data)
y = x**2 - 2 * x + 1
print(y.data)
y.backward()
print(x.grad)
x.grad * 0.01
y = x**2 - 2 * x + 1
print(y)
