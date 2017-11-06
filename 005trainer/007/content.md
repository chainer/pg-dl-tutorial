# 6. Save the trained model

Chainer provides two types of [serializers](https://docs.chainer.org/en/latest/reference/serializers.html#module-chainer.serializers) that can be used to save and restore model state. One supports the HDF5 format and the other supports the NumPy NPZ format. For this example, we are going to use the NPZ format to save our model since it is easy to use with NumPy and doesn’t need to install any additional dependencies or libraries.
