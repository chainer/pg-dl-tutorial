# Use Pre-trained Models

Various ways to write your models were described above. It turns out that VGG16 and ResNet are very useful as general feature extractors for many kinds of tasks, including but not limited to image classification. So, Chainer provides you with the pre-trained VGG16 and ResNet-50/101/152 models with a simple API. You can use these models as follows:

---

When [VGG16Layers](https://docs.chainer.org/en/latest/reference/generated/chainer.links.VGG16Layers.html#chainer.links.VGG16Layers) is instantiated, the pre-trained parameters are automatically downloaded from the author’s server. So you can immediately start to use VGG16 with pre-trained weight as a good image feature extractor. See the details of this model here: [chainer.links.VGG16Layers](https://docs.chainer.org/en/latest/reference/generated/chainer.links.VGG16Layers.html#chainer.links.VGG16Layers).

In the case of ResNet models, there are three variations differing in the number of layers. We have `chainer.links.ResNet50`, `chainer.links.ResNet101`, and `chainer.links.ResNet152` models with easy parameter loading feature. ResNet’s pre-trained parameters are not available for direct downloading, so you need to download the weight from the author’s web page first, and then place it into the dir `$CHAINER_DATSET_ROOT/pfnet/chainer/models` or your favorite place. Once the preparation is finished, the usage is the same as VGG16:

```python
from chainer.links import ResNet152Layers

model = ResNet152layers()
```

Please see the details of usage and how to prepare the pre-trained weights for ResNet here: [chainer.links.ResNet50](https://docs.chainer.org/en/latest/reference/generated/chainer.links.ResNet50Layers.html#chainer.links.ResNet50Layers)
