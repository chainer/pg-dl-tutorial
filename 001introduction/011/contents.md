# シリアライザ

シリアライザは現在の状態をストレージなどに保存したり，読み込んだりする機能です。
`Link`, `Optimizer`, `Trainer` がシリアリザをサポートしています。

```
serializers.save_npz('my.model', model)
```

これは，modelを'my.model'にNPZ（numpy + zip）形式で保存します。

保存されたモデルは `load_npz` で読み込むことができます。

```
serializers.load_npz'my.model', model)
```

同様に，HDF5フォーマットで保存するための `save_hdf5`, `load_hdf5` が存在します。

なお，シリアライズされるのは，parametersとpersistent valuesのみでそれ以外の属性値はシリアライズされないことに注意してください。シリアライズの対象にするには，`add_persistent()` を利用してください。


## 拡張機能

Trainerは拡張機能をサポートしています。
以下に代表的な拡張機能を紹介していきます。

以下では

```
from chainer.training import extentions
```

として拡張機能が既に `import` されているものとします。

## `Observations`, `Reporter`

拡張機能を説明する前に， `Observations` と `Reporter` という仕組みを紹介します。
ユーザーが監視した値を集めるための機能として `Reporter` があります。
`Reporter` は値の名前と実際の値のマッピングを保持します。
このマッピングを `Observations` とよびます。

例えば， `"accuracy"` という値の名前について，実際の値"0.975"が `Reporter` は登録しているとします。

Chainerの中で層やネットワークに対応する `Link` や `Chain` はこれらの `Reporter` の機能を備えています。


## `Evaluator`

`Evaluator` は学習が終わった後に， `test_iter` で定義されるテストデータセットで評価をします。

```
trainer.extend(extensions.Evaluator(test_iter, model))
```

## `LogReport`

`LogReport` は `reporter` に報告された値をログファイルに格納してくれます。

```
trainer.extend(extensions.LogReport())
```

`PrintReport` は指定した値名を持つ `Observations` を表示してくれます。

```
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
```

`ProgressBar` は学習の進捗度合いをプログレスバーで表示してくれます。

```
trainer.extend(extensions.ProgressBar())
```

`Snapshot` は定期的にモデルのスナップショットを記録し，出力ディレクトリに格納します。

```
trainer.extend(extensions.Snapshot((10, 'epoch')))
```
