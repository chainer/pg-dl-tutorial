# Let’s try using the Trainer feature

By using [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer), you don’t need to write the tedious training loop explicitly any more. Furthermore, Chainer provides many useful extensions that can be used with [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) to visualize your results, evaluate your model, store and manage log files more easily.

This example will show how to use the [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) to train a fully-connected feed-forward neural network on the MNIST dataset.

**Note**

If you would like to know how to write a training loop without using this functionality, please check [How to write a training loop in Chainer](training_loop_in_chainer.ipynb) instead of this tutorial.
