# 3. Define a network

Now let’s define a neural network that we will train to classify the MNIST images. For simplicity, we use a three-layer perceptron here. We set each hidden layer to have 100 units and set the output layer to have 10 units, which is corresponding to the number of class labels of the MNIST.

## Create your network as a subclass of Chain

You can create your network by writing a new subclass of [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain). The main steps are twofold:

1. Register the network components which have trainable parameters to the subclass. Each of them must be instantiated and assigned to a property in the scope specified by [init_scope()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain.init_scope)
2. Define a `__call__()` method that represents the actual **forward computation** of your network. This method takes one or more [Variable](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Variable.html#chainer.Variable), `numpy.array`, or `cupy.array` as its inputs and calculates the forward pass using them.

---

[Link](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Link.html#chainer.Link), [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain), [ChainList](https://docs.chainer.org/en/latest/reference/core/generated/chainer.ChainList.html#chainer.ChainList), and those subclass objects which contain trainable parameters should be registered to the model by assigning it as a property inside the [init_scope()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain.init_scope). For example, a [Function](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Function.html#chainer.Function) does not contain any trainable parameters, so there is no need to keep the object as a property of your network. When you want to use [relu()](https://docs.chainer.org/en/latest/reference/generated/chainer.functions.relu.html#chainer.functions.relu) in your network, using it as a function in `__call__()` works correctly.

In Chainer, the Python code that implements the forward computation itself represents the network. In other words, we can conceptually think of the computation graph for our network being constructed dynamically as this forward computation code executes. This allows Chainer to describe networks in which different computations can be performed in each iteration, such as branched networks, intuitively and with a high degree of flexibility. This is the key feature of Chainer that we call **Define-by-Run**.
