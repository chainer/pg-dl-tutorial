# Chainerを使った深層学習 MNIST

それではさっそく，Chainerを使った深層学習を動かしてみましょう。

ここでは，MNISTとよばれる手書き文字データセットを使って，多層パーセプトロン（multilayer perceptron, MLP）による多クラス分類器の学習をしてみましょう。
MNISTデータセットは70000枚の28*28のグレイスケール画像から構成されており，それぞれに0〜9の数字がかかれています。
このデータセットを60000の学習データと，10000のテストデータに分けて使います。

なお，以降では各画像を28*28の画素を並べた784のグレイスケール値がならんた784次元のベクトルとして扱うようにします。

MNISTデータセットのダウンロードは次の `dataset.get_mnist()` を呼び出すことで実行されます。

```
from chainer import dataset
train, test = dataset.get_mnist()
```

これらのデータは `chainer.TupleDataset` で構成されており，各サンプルが画像とそのラベル（0〜9）のタプルから構成されています。
例えば，train[100]は100番目のデータの画像とラベルからなるタプルを返します

```
x, y = train[100]
print x
print y
```

xは784次元のベクトル，yがラベル（整数値）です。
なお， `print x` はChainer Playgroundでは大きすぎてそのままでは表示できませんし，
単純な数値として表示されるだけなのでよく分かりません。

そのためChainer Playgroundでは専用の補助関数 `print_mnist` を用意しています。
それを利用することでMNISTデータセットの画像を表示できます。
`print_mnist` を使用するためにはまず `playground` をインポートします。

```
import playground
```

その後，

```
x, y = train[100]
playground.print_mnist(x)
print y
```

とすることで100番目のデータを画像として表示します。

## 課題

trainの各ラベル毎の画像の平均ベクトルを求め，それらを順に表示せよ

## 補足： `dataset.get_mnist()` によるデータセットの取得

`dataset.get_mnist()` は一度目の呼び出し時は実際にデータセットをダウンロードするため遅いですが，
二回目以降は既にダウンロードされているキャッシュを利用して実行されますので速く処理されます。手元で `dataset.get_mnist()` を実行する際，ダウンロード先は環境変数
`CHAINER_DATASET_ROOT` で指定することができます。
デフォルトは `~/.chainer/dataset` です。

また，Chainer Playground内では事前にデータセットをダウンロードしキャッシュ済みの状態になっています。
そのためChainer Playgroundでは何度 `dataset.get_mnist()` を実行してもデータセットを公開しているサーバに負担がかかることはありません。

## 補足：他のデータセット

MNISTは，様々な研究開発のベースラインとしてよく使用されているデータセットであり，新しい手法の性能評価，デバッグのためにも使われます。
最近は，より現実世界の問題に近い複雑なデータセットであるCIFAR-10，CIFAR-100やSVHN（Street View House Numbers)データセットが使われる場合も多くなっています。
http://ufldl.stanford.edu/housenumbers/
