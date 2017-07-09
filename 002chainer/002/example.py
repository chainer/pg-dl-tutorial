from chainer import Variable
import numpy as np

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = x ** 2 - 2 * x + 1
y.backward()

z = Variable(np.array([10, 20], dtype=np.float32))
zz = 2 * z
zz.grad = np.array([0.1, -0.1], dtype=np.float32)
zz.backward()

print(x.grad)
print(z.grad)

# print(y.data)
