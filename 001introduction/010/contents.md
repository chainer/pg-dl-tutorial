# MNIST：Iterator, Updater, Trainer

Chainerでは学習操作を抽象化するために，`Iterator`, `Updater`, `Optimizer`, `Trainer` の四つの機能を備えています。

これらを順に紹介していきます。

## `Iterator`

`Iterator` はデータセット上の操作，アクセスを抽象化します。
構築時にデータセットを引数として指定すると，そのデータセットに対する `Iterator` を返します。
引数として，`batch_size` は，一度のアクセスでいくつ同時に読み込むか， `shuffleは` アクセスの際にランダムにアクセスするかどうかを指定します。

```
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
```

## `Optimizer`

`Optimizer` はパラメータの最適化を担当します。
ここでは複数ある `Optimizer` の中で `Adam` を使います。
`Adam` は広い学習問題で安定して学習できるモデルです。

`Optimizer` は最適化対象の `Link` を指定する必要があります。

```
opt = chainer.optimizers.Adam()
opt.setup(model)
```

## `Updater`

`Updater` はパラメータの更新を担当します。
`Updater` にはこれまでに作った `Iterator` と `Optimizer` を渡します。

```
updater = training.StandardUpdater(train_iter, opt)
```

## `Trainer`

`Trainer` は訓練の実行を担当します。
`Trainer` の構築時には， `Updater` と学習の回数，そして出力結果を指定します。

```
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
```

二つ目の引数は訓練回数を示す引数であり，単位として `'epoch'` か `'iteration'` を受けとります。

例えば， `(20, 'epoch')` はデータ全体を20回走査するという意味ですし， `(1000, 'iteration')` はミニバッチを1000回動かすという意味です。

これで準備ができました。後は `run()` をよびだし実行するだけです。

```
trainer.run()
```

## 課題

`Trainer` を実際に動かし学習できることを確かめてください。
その上で例えばユニット数を変えたり，収束回数を変えたり， `Optimizer` を変えたりして精度が変わることを確認してください。
