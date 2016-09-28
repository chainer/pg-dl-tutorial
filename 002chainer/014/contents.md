# Chainerの基本：MNIST 例

実はこれまで紹介してきた多くの処理は実はUpdater, Trainerとよばれる仕組みを使えば
ユーザーが書く必要はありません。

これまでは，内部でどのような処理をしているのかを知ってもらうためにあえて説明をしました。

それではUpdater, Trainerの機能を使って学習部分を書き直していきましょう。

まず，updateを担当するUpdaterを用意します

```
updater = training.StandardUpdater(train_iter, optimizer)
```

updaterには既に作成したiteratorとoptimizerを渡します。
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

先程はepochの回数だけループを回し，その中でbackwardを呼び出したりといろいろな手間がかかっていましたが
今回はUpdater, Trainerを用紙するだけで実現できたことに注意してください。

さらに，学習の際に必要な機能の多くはextentionsとよばれる拡張機能により実現することができます。

例えば，次のような拡張機能がよく使われます。

* Evaluatorは学習が終わった後に，test_iterで定義されるテストデータセットで評価をしてくれます

```
trainer.extend(extensions.Evaluator(test_iter, model))
```

* LogReportは報告された値をlog fileに格納してくれます。
```
trainer.extend(extensions.LogReport())
```

* PrintReportは指定したカラムを表示してくれます。
```
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
```

* ProgressBarは進捗度合いをプログレスバーで表示してくれます。
```
trainer.extend(extensions.ProgressBar())
```

* Snapshotは定期的にモデルのスナップショットを記録し，出力ディレクトリに格納します。
```
trainer.extend(extensions.Snapshot((10, 'epoch')))
```

## 課題

trainerを実際に動かし学習できることを確かめてください。
その上で例えばユニット数を変えたり，収束回数を変えたり，Optimizerを変えたりして精度が変わることを
確認してください。





