2.3 Generating sentences
------------------------

You can generate the sentence which starts with a word in the vocabulary. In this example, we generate a sentence which starts with the word **apple**. We use the script in the PTB example of the official repository.

https://github.com/chainer/chainer/tree/master/examples/ptb

```bash
python gentxt.py -m ptb_result/model_epoch_39 -p apple
```

#### Output example:

```
apple <unk> is the major public <unk> <unk> business in N years .this is a regime of the earth as
```
