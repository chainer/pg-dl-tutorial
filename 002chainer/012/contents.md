# Chainerの基本：MNIST 例

次にネットワークアーキテクチャを定義します。

次の例は，3層からなるニューラルネットワークであり，中間層のユニット数がn_unitsであり，出力層のunit数からなります。

この際，Linearの入力サイズがNoneとなっていますがこれは最初の実行時に入力から推論されます。

```
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

次に，損失関数を定義するClassifierを定義します。
Classiferは精度を計算した上で損失をsoftmax_cross_entropyを使って定義します。

```
class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
```

これとほぼ同じ機能が既にchainer.links.Classifierで実装されています。

```
model = L.Classifier(MLP(784, 100, 10))
opt = optimizers.Adam()
opt.setup(model)
```

なお，L.Classiferは初期化時に次の三つの関数を受け取れるようになっています

* predictor
　学習対象であるLink
* lossfun
　誤差関数に使う関数。上記例の場合はF.softmax_cross_entropy
* accfun
　精度評価につかう関数。上記の場合はF.accuracy

## 課題

右のコードで各ミニバッチに対し，MLPで予測した結果のaccuracyを表示しなさい．

