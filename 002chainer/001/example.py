import numpy as np
from chainer import Variable

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = x**2 - 2 * x + 1
print(y.data)

z_data = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
z = Variable(z_data)
print(z[:, 1].data)  # [[3], [6]]
