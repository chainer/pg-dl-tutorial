# シリアライザ

シリアライザは現在の状態をディスクなどに保存したり，読み込んだりする
機能です。
Link, Optimizer, Trainerがシリアリザをサポートしています。

```
serializers.save_npz('my.model', model)
```

これは，modelを'my.model'にNPZ（numpy + zip）形式で保存します。

保存されたモデルはload_npzで読み込むことができます。

```
serializers.load_npz'my.model', model)
```

同様に，HDF5フォーマットで保存するためのsave_hdf5, load_hdf5が存在します。

なお，シリアライズされるのは，parametersとpersistent valuesのみでそれ以外の属性値はシリアライズされないことに注意してください。シリアライズの対象にするには，add_persistent()を利用してください。


# 拡張機能

Trainerは拡張機能をサポートしています。
以下に代表的な拡張機能を紹介していきます。

以下では

```
from chainer.training import extentions
```

として拡張機能が既にimportされているものとします。

## Observations, Reporter

拡張機能を説明する前に，ObservationsとReporterという仕組みを紹介します。
ユーザーが監視した値を集めるための機能としてReporterがあります。
Reporterは値の名前と実際の値のマッピングを保持します。
このマッピングをObservationsとよびます。

例えば，"accuracy"という値の名前について，実際の値"0.975"がReporterは登録しているとします。

Chainerの中で層やネットワークに対応するLinkやChainはこれらのReporterの機能を備えています。



## Evaluator

Evaluatorは学習が終わった後に，test_iterで定義されるテストデータセットで評価をします。

```
trainer.extend(extensions.Evaluator(test_iter, model))
```

## logReport

LogReportはreporterに報告された値をlog fileに格納してくれます。

```
trainer.extend(extensions.LogReport())
```

* PrintReportは指定した値名を持つobservationsを表示してくれます。
```
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
```

## ProgressBarは学習の進捗度合いをプログレスバーで表示してくれます。
```
trainer.extend(extensions.ProgressBar())
```

## Snapshotは定期的にモデルのスナップショットを記録し，出力ディレクトリに格納します。
```
trainer.extend(extensions.Snapshot((10, 'epoch')))
```


