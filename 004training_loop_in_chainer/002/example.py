from chainer.datasets import mnist

# Download the MNIST data if you haven't downloaded it yet
train, test = mnist.get_mnist(withlabel=True, ndim=1)

# Display an example from the MNIST dataset.
# `x` contains the inpu t image array and `t` contains that target class
# label as an integer.
x, t = train[0]
print('label:', t)
