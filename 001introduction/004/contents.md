# ネットワークの定義

次にネットワークを定義します。
Chainerの特徴は"Define by Run"，つまり順に実行しながらネットワークを定義していくというものです。

以下に3層のニューラルネットワークの例をあげます。
ネットワークの作り方は，chainer.Chainを継承した上で，コンストラクタで
利用するパラメータを定義します。
この他の定義の仕方についてはこれから紹介していきます。

このMLPオブジェクトは関数呼び出しでどのような計算をするかを定義します。
例えば，self.l1というのは初期化で定義したl1=L.Linearを使うということを意味しています。
この計算の中でChainerは自動的に学習に必要な情報（計算グラフ）を作成していきます。

```
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

このように作ったパーセプトロンを分類器として使う場合には以下のようにL.Classifierを挟んでモデルを作ります。
Classifierはデフォルトでは分類器softmax，学習時の損失関数はsoftmaxクロスエントロピー損失を使います。
```
model = L.Classifier(MLP(784, 10))
```


## 課題

Chainで継承されたオブジェクトに登録されているパラメータ付関数はnamed_linksで呼び出すことができます。
例えば上の例の場合l1, l2が呼び出されます。
上の例を4層のニューラルネットワークに変更し，そのnamed_linksを表示してください。



