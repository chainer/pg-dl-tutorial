# How to write a training loop in Chainer

In this tutorial section, we will learn how to train a deep neural network to classify images of hand-written digits in the popular MNIST dataset. This dataset contains 50,000 training examples and 10,000 test examples. Each example is a set of a 28 x 28 greyscale image and a corresponding class label. Since the digits from 0 to 9 are used, there are 10 classes for the labels.

Chainer provides a feature called [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) that can simplify the training procedure of your model. However, it is also good to know how the training works in Chainer before starting to use the useful [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) class that hides the actual processes. Writing your own training loop can be useful for learning how [Trainer](https://docs.chainer.org/en/latest/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) works or for implementing features not included in the standard trainer.

The complete training procedure consists of the following steps:

1. Prepare a dataset
2. Create a dataset iterator
3. Define a network
4. Select an optimization algorithm
5. Write a training loop
    1. Retrieve a set of examples (mini-batch) from the training dataset.
    2. Feed the mini-batch to your network.
    3. Run a forward pass of the network and compute the loss.
    4. Just call the [backward()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Variable.html#chainer.Variable.backward) method from the loss [Variable](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Variable.html#chainer.Variable) to compute the gradients for all trainable parameters.
    5. Run the optimizer to update those parameters.
6. Save the trained model
7. Perform classification by the saved model and check the network performance on validation/test sets.
