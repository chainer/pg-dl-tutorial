# Trainerの拡張機能

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



