# 2. Create a dataset iterator

Although this is an optional step, we’d like to introduce the [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) class that retrieves a set of data and labels from the given dataset to easily make a mini-batch. There are some subclasses that can perform the same thing in different ways, e.g., using multi-processing to parallelize the data loading part, etc.

Here, we use [SerialIterator](https://docs.chainer.org/en/latest/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator), which is also a subclass of [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) in the example code below. The [SerialIterator](https://docs.chainer.org/en/latest/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator) can provide mini-batches with or without shuffling the order of data in the given dataset.

All [Iterators](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) produce a new mini-batch by calling its [next()](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator.next) method. All [Iterators](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) also have properties to know how many times we have taken all the data from the given dataset (epoch) and whether the next mini-batch will be the start of a new epoch (`is_new_epoch`), and so on.

The code below shows how to create a [SerialIterator](https://docs.chainer.org/en/latest/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator) object from a dataset object.

**Note**

`iterator`s can take a built-in Python list as a given dataset. It means that the example code below is able to work,

```
train = [(x1, t1), (x2, t2), ...]  # A list of tuples
train_iter = iterators.SerialIterator(train, batchsize)
```

where `x1, x2, ...` denote the input data and `t1, t2, ...` denote the corresponding labels.

## Details of [SerialIterator](https://docs.chainer.org/en/latest/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator)

- [SerialIterator](https://docs.chainer.org/en/latest/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator) is a built-in subclass of [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator) that can retrieve a mini-batch from a given dataset in either sequential or shuffled order.
- The [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator)‘s constructor takes two arguments: a dataset object and a mini-batch size.
- If you want to use the same dataset repeatedly during the training process, set the `repeat` argument to `True` (default). Otherwise, the dataset will be used only one time. The latter case is actually for the evaluation.
- If you want to shuffle the training dataset every epoch, set the `shuffle` argument to `True`. Otherwise, the order of each data retrieved from the dataset will be always the same at each epoch.

In the example code shown above, we set `batchsize = 128` in both `train_iter` and `test_iter`. So, these iterators will provide 128 images and corresponding labels at a time.
