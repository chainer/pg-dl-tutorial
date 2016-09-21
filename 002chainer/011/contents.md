# Chainerの基本：MNIST 例

ここまで紹介してきたのを元に，多層パーセプトロンによる他クラス分類タスクの学習をしてみましょう。

ここでは，MNISTとよばれる手書き文字データセットを対象にします。

Memo:
MNISTは，様々な研究開発のベースラインとしてよく使用されているデータセットであり，新しい手法の性能評価，デバッグのためにも使われます。
より現実世界の問題に近い複雑なデータセットであるSVHN（Street View House Numbers)データセットが使われる場合も多くなっています。
http://ufldl.stanford.edu/housenumbers/

MNISTデータセットは70000枚の28*28のグレイスケール画像から構成されており，それぞれに0〜9の数字がかかれています。

このデータセットを60000の学習データと，10000のテストデータに分けます。

なお，以降では各画像を28*28の画素を並べた784の画素値がならんた784次元のベクトルとして扱うようにします

MNISTデータセットはダウンロードしてくるスクリプトが用意されています。

```
train, test = dataset.get_mnist()
```

これらはTupleDatasetで構成されており，各サンプルが画像とそのラベル（0〜9）のタプルから構成されています。

（著者注意）全員ダウンロードすると負荷がすごいかかるのでキャッシュ先から読むとか何かしら対策必要


次に，このデータセット上を走査するIteratorを用意します。

```
train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)
```

testの場合はシャッフルは必要なにのでshuffle=False, 繰り返しも必要ないのでrepeat=Falseとしています。

これを使うには例えば，次のようにするとbatchにはbatchsize分だけのデータが読み込まれています。

```
train_num = len(train)
batchsize = 100
for i in xrange(0, train_num, batchsize):
  batch = train_iter.next()
```


## 課題

trainのデータセットの各次元の平均meanを求め，それをtrainから引いた上で，trainの各次元の平均が0になることを確認せよ。
