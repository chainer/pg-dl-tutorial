# MNISTの学習

これまで，学習/評価用データセットを作り，また学習対象のモデルを作りました。

```
train_full, test_full = chainer.datasets.get_mnist()
train = datasets.SubDataset(train_full, 0, 1000)
test = datasets.SubDataset(test_full, 0, 1000)
model = L.Classifier(MLP(100, 10))
```

それでは実際に学習させてみましょう。
今回は時間短縮のためMNISTデータセットのうち1000件のみを用いることにします。

Chainerでは学習操作を抽象化するための機能が揃っています。
これらを利用することで殆ど自分でコードを書くことなく学習させることができます。

はじめにデータセット上の操作を抽象化する `Iterator` を用意します。
`Iterator` は構築時にデータセットを引数として指定すると，そのデータセットに対する `Iterator` を返します。
引数として，`batch_size` は，一度のアクセスでいくつ同時に読み込むか， `shuffle` はアクセスの際にランダムにアクセスするかどうかを指定します。

```
# Set up a iterator
batchsize = 100
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                             repeat=False, shuffle=False)
```

次に，パラメータの最適化を担当する `Optimizer` を用意しします。
ここでは複数ある `Optimizer` の中で `Adam` を使います。
`Adam` は広い学習問題で安定して学習できる手法です。

`Optimizer` はsetup()で最適化対象の `Chain` または `Link` を指定する必要があります。

```                                
# Set up an optimizer
opt = chainer.optimizers.Adam()
opt.setup(model)
```

次に，実際のパラメータ更新を担当する `Updater` を用意します。
これまで用意した学習用データに対するIterator，最適化を担当する `Optimizer` ，そしてどのデバイスで
実行するのかを指定します。`device=-1`はCPUを使うことを表します。

```
# Set up an updater
updater = training.StandardUpdater(train_iter, opt, device=-1)
```

最後に学習ループを担当する `Trainer` を用意します。
今回は5エポック(5回データセットを走査する)だけ学習を回すようにします。

```
# Set up a trainer
epoch = 5
trainer = training.Trainer(updater, (epoch, 'epoch'), out='/tmp/result')
```

`Trainer` は様々な拡張機能を使うことができます。

評価データで評価をするには，次のようにします。

```
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
```

学習途中の結果を表示するには，次のようにします。
1エポックごとに、trainデータに対するaccuracyと、testデータに対するaccuracyを出力させます。

```
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport(
        ['epoch', 
         'main/accuracy', 'validation/main/accuracy']), trigger=(1, "epoch"))
```

これで全て用意ができました。
trainerのrunを呼び出すことで学習できます。

```
# Run the trainer
trainer.run()
```

最後に学習の結果を確認してみましょう。
ランダムに選んだテストデータ一件に対する予測を出力します。

```
x, y = test[np.random.randint(len(test))]
playground.print_mnist(x)
pred = F.softmax(model.predictor(Variable(x.reshape((1, 784))))).data
print "Prediction: ", np.argmax(pred)
print "Correct answer: ", y
```


## 課題

`Trainer` を実際に動かし学習できることを確かめてください。
その上で例えばユニット数を変えたり，epoch（学習回数）を変えたり， `Optimizer` を `RMSprop()` などに変えたりして精度が変わることを確認してください。
