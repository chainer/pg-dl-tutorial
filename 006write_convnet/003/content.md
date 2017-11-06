### Ways to calculate loss

When you train the model with label vector `t`, the loss should be calculated using the output from the model. There also are several ways to calculate the loss:

---

This is a primitive way to calculate a loss value from the output of the model. On the other hand, the loss computation can be included in the model itself by wrapping the model object ([Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) or [ChainList](https://docs.chainer.org/en/latest/reference/core/generated/chainer.ChainList.html#chainer.ChainList) object) with a class inherited from Chain. The outer [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) should take the model defined above and register it with [init_scope()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain.init_scope). [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) is actually inherited from Link, so that [Chain](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Chain.html#chainer.Chain) itself can also be registered as a trainable [Link](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Link.html#chainer.Link) to another Chain. Actually, [Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier) class to wrap the model and add the loss computation to the model already exists. Actually, there is already a [Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier) class that can be used to wrap the model and include the loss computation as well. It can be used like this:

```python
model = L.Classifier(LeNet5())

# Foward & Loss calculation
loss = model(x, t)
```

This class takes a model object as an input argument and registers it to a `predictor` property as a trained parameter. As shown above, the returned object can then be called like a function in which we pass `x` and `t` as the input arguments and the resulting loss value (which we recall is a [Variable](https://docs.chainer.org/en/latest/reference/core/generated/chainer.Variable.html#chainer.Variable)) is returned.

See the detailed implementation of [Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier) from here: [chainer.links.Classifier](https://docs.chainer.org/en/latest/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier) and check the implementation by looking at the source.

From the above examples, we can see that Chainer provides the flexibility to write our original network in many different ways. Such flexibility intends to make it intuitive for users to design new and complex models.
