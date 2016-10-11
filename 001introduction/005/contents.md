# MNIST：Iterator, Updaterr, Trainer

Chainerでは学習操作を抽象化するために，Iterator, Updater, Optimizer, Trainerの四つの機能を備えています。

これらを順に紹介していきます。

## Iterator

Iteratorはデータセット上の操作，アクセスを抽象化します。
構築時にデータセットを引数として指定すると，そのデータセットに対するIteratorを返します。
引数として，batch_sizeは，一度のアクセスでいくつ同時に読み込むか，shuffleはアクセスの際にランダムにアクセスするかどうかを指定します。

```
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
```

## Optimizer

Optimizerはパラメータの最適化を担当します。
ここでは複数あるOptimizerの中でAdamを使います。
Adamは広い学習問題で安定して学習できるモデルです。

Optimizerは最適化対象のLinkを指定する必要があります。

```
opt = chainer.optimizers.Adam()
opt.setup(model)
```

## Updater

Updaterはパラメータの更新を担当します。
Updaterにはこれまでに作ったIteratorとOptimizerを渡します。

```
updater = training.StandardUpdater(train_iter, opt)
```

## Trainer

Trainerは訓練の実行を担当します。
Trainerの構築時には，Updaterと学習の回数，そして出力結果を指定します。
```
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
```

二つ目の引数は訓練回数を示す引数であり，単位として'epoch'か'iteration'を受けとります。

例えば，(20, 'epoch')はデータ全体を20回走査するという意味ですし，('1000', 'iteration'）はミニバッチを1000回動かすという意味です。

これで準備ができました。後はrun()をよびだし実行するだけです。

```
trainer.run()
```

## 課題

trainerを実際に動かし学習できることを確かめてください。
その上で例えばユニット数を変えたり，収束回数を変えたり，Optimizerを変えたりして精度が変わることを確認してください。

