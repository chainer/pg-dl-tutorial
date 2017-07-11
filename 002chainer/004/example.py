from chainer import functions as F
from chainer import Variable
import numpy as np


x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = F.relu(x)

z = Variable(np.array([[10, 20], [30, 40]], dtype=np.float32))
zz = F.transpose(z)
print(zz.data)

# print() exp(x)+sin(x)
