### 2.2.9 Create RNN and classification model

Instantiate RNNLM model and wrap it with `L.Classifier` because it calculates softmax cross entropy as the loss.

Note that `chainer.links.Classifier` computes not only the loss but also accuracy based on a given input/label pair. To learn the RNN language model, we only need the loss (cross entropy) in the `Classifier` because we calculate the perplexity instead of classification accuracy to check the performance of the model. So, we turn off computing the accuracy by giving `False` to `model.compute_accuracy` attribute.
