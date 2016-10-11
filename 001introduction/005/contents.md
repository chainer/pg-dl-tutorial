# MNIST：Iterator, Updaterr, Trainer

Chainerでは学習操作を抽象化するために，Iterator, Updater, Optimizer, Trainerの四つの役割を組み合わせます。

Iteratorはデータセット上の操作，アクセスを抽象化します。

データセットを引数にとりどのようにアクセスするかをオプションで指定します。
これらのオプションについてはまた後で説明します。

```
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
```

次にパラメータの最適化を担当するOptimizerを用意します。
ここでは複数あるOptimizerの中でAdamを使います。

```
opt = chainer.optimizers.Adam()
opt.setup(model)
```

次にパラメータの更新を担当するUpdaterを用意します。
Updaterには作ったIteratorとOptimizerを渡します。

```
updater = training.StandardUpdater(train_iter, opt)
```

次に訓練を実行するTrainerをupdaterを渡して作ります。

```
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
```

二つ目の引数は訓練回数を示す引数であり，単位として'epoch'か'iteration'を受けとります。

例えば，(20, 'epoch')はデータ全体を20回走査するという意味ですし，('1000', 'iteration'）はミニバッチを1000回動かすという意味です。

準備ができました。後はrun()をよびだし実行するだけです。

```
trainer.run()
```

## 課題

trainerを実際に動かし学習できることを確かめてください。
その上で例えばユニット数を変えたり，収束回数を変えたり，Optimizerを変えたりして精度が変わることを
確認してください。

