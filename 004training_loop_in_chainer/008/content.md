# 7. Perform classification by the saved model

Let’s use the saved model to classify a new image. In order to load the trained model parameters, we need to perform the following two steps:

1. Instantiate the same network as what you trained.
2. Overwrite all parameters in the model instance with the saved weights using the [load_npz()](https://docs.chainer.org/en/latest/reference/generated/chainer.serializers.load_npz.html#chainer.serializers.load_npz) function.

Once the model is restored, it can be used to predict image labels on new input data.
