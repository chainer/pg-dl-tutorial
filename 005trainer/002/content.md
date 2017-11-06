# 1. Prepare the dataset

Load the MNIST dataset, which contains a training set of images and class labels as well as a corresponding test set.

**Note**
You can use a **Python list as a dataset**. Because all types of objects whose element can be accessed via `[]` accessor and lengh can be obtained with `len()` function, can be used as a dataset given to the [Iterator](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.Iterator.html#chainer.dataset.Iterator). For example,

```
train = [(x1, t1), (x2, t2), ...]
```

a list of tuples like this can also be used equally to a [DatasetMixin](https://docs.chainer.org/en/latest/reference/core/generated/chainer.dataset.DatasetMixin.html#chainer.dataset.DatasetMixin) object.

But many useful abstracted [datasets](https://docs.chainer.org/en/latest/reference/datasets.html#module-chainer.datasets) enable to avoid storing all data on the memory at a time, so it’s better to use them for large datasets. For example, [ImageDataset](https://docs.chainer.org/en/latest/reference/generated/chainer.datasets.ImageDataset.html#chainer.datasets.ImageDataset) takes paths to image files as its argument, and just keep the list in the dataset object. It means the actual image data will be loaded from disks using given paths when [\_\_getitem\_\_()](https://docs.chainer.org/en/latest/reference/generated/chainer.datasets.ImageDataset.html#chainer.datasets.ImageDataset.__getitem__) is called. Until then, no images are loaded to the memory, so it can save the memory consumption.
