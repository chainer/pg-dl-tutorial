# Chainerの基本：MNIST 例

ここまで紹介してきた多くの処理は実はUpdater, Trainerとよばれる仕組みを使えば
ユーザーが書く必要はありません。

これまでは，内部でどのような処理をしているのかを知ってもらうためにあえて順に説明をしました。

それでは実際に使って置き換えていきましょう。

まず，updateを担当するUpdaterを用意します

```
updater = training.StandardUpdater(train_iter, optimizer)
```

updaterには既に作成したiteratorとoptimizerを渡します。
次に実際に訓練を実行するtrainerを作ります。

```
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
```

二つ目の引数はどれだけ訓練するかを示す引数であり，'epoch'か'iteration'のどちらかを単位として使うことができます。

例えば，(20, 'epoch')はデータ全体を20回走査するという意味ですし，('1000', 'iteration'）はミニバッチを1000回動かすという意味です。

準備ができました。後はrun()をよびだし実行するだけです。

```
trainer.run()
```

さて，訓練の途中でどのように学習が進んでいるかを調べたい場合があります。
この場合、run()を呼び出す前に次のような拡張機能を登録しておきます。

```
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.Snapshot((10, 'epoch')))
```

Evaluatorは学習が終わった後に，test_iterで定義されるテストデータセットで評価をしてくれます
LogReportは報告された値をlog fileに格納してくれます。
PrintReportは指定したカラムを表示してくれます。
ProgressBarは進捗度合いをプログレスバーで表示してくれます。
Snapshotは定期的にモデルのスナップショットを記録し，出力ディレクトリに格納します。





