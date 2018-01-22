### 2.2.3 Define Network Structure

An RNNLM written in Chainer is shown below. It implements the model depicted in the above figure.

- When we insatantiate this class for making a model, we give the vocavulary size to `n_vocab` and the size of hidden vectors to `n_units`.
- This network uses `chainer.links.LSTM`, `chainer.links.Linear`, and `chainer.functions.dropout` as its building blocks.
- All the layers are registered and initialized in the context with `self.init_scope()`.
- You can access all the parameters in those layers by calling `self.params()`.
- In the constructor, it initializes all parameters with values sampled from a uniform distribution $U(-1, 1)$.
- The `__call__` method takes an word ID `x`, and calculates the word probability vector for the next word by forwarding it through the nerwork, and returns the output.
- Note that the word ID `x` is automatically converted to a $|\mathcal{V}|$-dimensional one-hot vector and then multiplied with the input embedding matrix in `self.embed(x)` to obtain an embed vector `h0` at the first line of `__call__`.
