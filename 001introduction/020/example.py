from chainer import datasets
import playground


train, test = datasets.get_mnist()
x, y = train[100]
playground.print_mnist(x)
print(y)
