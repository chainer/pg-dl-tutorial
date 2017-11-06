### 2.2.4 Load the Penn Tree Bank long word sequence dataset

In this notebook, we use [Penn Tree Bank](https://www.cis.upenn.edu/~treebank/) dataset that contains number of sentences. Chainer provides an utility function to obtain this dataset from server and convert it to a long single sequence of word IDs. `chainer.datasets.get_ptb_words()` actually returns three separated datasets which are for train, validation, and test.

Let's download and make dataset objects using it:
