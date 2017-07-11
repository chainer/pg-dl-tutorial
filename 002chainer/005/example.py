from chainer import links as L
import numpy as np


f = L.Linear(3, 2)
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]],
             dtype=np.float32)
y = f(x)
print(y.data)
