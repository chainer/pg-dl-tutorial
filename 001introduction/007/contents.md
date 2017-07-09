# 1. 学習対象のモデルを定義する (4)

これまで扱ったLinkとFunctionを組み合わせて，学習対象のモデルを実際に作ってみましょう。

以下に三層からなるニューラルネットワークの例をあげます。

```
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

各機能は今後詳細に説明されますので，ここでは概要だけ説明します。
詳細は理解できなくてもそのまま飛ばして問題ありません。

このMLPは，三つのLinear（l1, l2, l3）を学習可能なパラメータとして持ち，`__call__`でそれらのパラメータを利用して結果を計算します。

なお，`L.Linear` の第一引数には `None` を指定することで実際の入力からユニット数を自動で設定してくれます。

`__call__` では先ほど定義した層に入力`x`を与えて計算（順計算）を行います。
まず `l1` に大元の入力 `x` を与え，それをLinearで変換したものにReLUを適用します。
その計算結果 `h1` を次の層 `l2` に与え同様の計算を行います。
`l3` に関しても同様に前層の結果を元に計算を行います。
最終的に `l3` の結果を返すことで計算が完了します。

このように，(1)Linkを使って学習対象のパラメータを定義し，(2)次にそれらを使って順計算を定義することで学習対象のモデルを定義できます。

