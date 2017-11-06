# 4. Select an optimization algorithm

Chainer provides a wide variety of optimization algorithms that can be used to optimize the network parameters during training. They are located in `optimizers` module.

Here, we are going to use the stochastic gradient descent (SGD) method with momentum, which is implemented by [MomentumSGD](https://docs.chainer.org/en/latest/reference/generated/chainer.optimizers.MomentumSGD.html#chainer.optimizers.MomentumSGD). To use the optimizer, we give the network object (typically it’s a [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) or [ChainList](https://docs.chainer.org/en/latest/reference/core/generated/chainer.ChainList.html#chainer.ChainList)) to the [setup()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Optimizer.html#chainer.Optimizer.setup) method of the optimizer object to register it. In this way, the [Optimizer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Optimizer.html#chainer.Optimizer) can automatically find the model parameters and update them during training.

You can easily try out other optimizers as well. Please test and observe the results of various optimizers. For example, you could try to change [MomentumSGD](https://docs.chainer.org/en/latest/reference/generated/chainer.optimizers.MomentumSGD.html#chainer.optimizers.MomentumSGD) to [Adam](https://docs.chainer.org/en/latest/reference/generated/chainer.optimizers.Adam.html#chainer.optimizers.Adam), [RMSprop](https://docs.chainer.org/en/latest/reference/generated/chainer.optimizers.RMSprop.html#chainer.optimizers.RMSprop), etc.

**Note**

In the above example, we set `lr` to 0.01 in the constructor. This value is known as the “learning rate”, one of the most important hyperparameters that need to be adjusted in order to obtain the best performance. The various optimizers may each have different hyperparameters and so be sure to check the documentation for the details.
