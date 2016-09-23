# Chainerの基本：Serializer

最後に紹介する機能がシリアライザです．
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
