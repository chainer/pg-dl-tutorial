import chainer
import numpy as np

print("Hello World!")
print(chainer.__version__)
print(355.0 / 113.0)
print(np.eye(5))

total = 0
for i in xrange(10):
    total += i

print(total)
