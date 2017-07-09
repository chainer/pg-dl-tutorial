from chainer import Variable
import numpy as np

x_data = np.array([2 * i + 1 for i in range(101)], dtype=np.float32)
x = Variable(x_data)
