# ネットワークの定義

次に学習対象のネットワークを定義します。

以下に3層のニューラルネットワークの例をあげます。

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

ネットワークを定義するオブジェクトはchainer.Chainを継承します。
そして，コンストラクタでネットワーク内で利用するパラメータを定義します。
この他のパラメータ定義の仕方についてはこれから紹介していきます。

このネットワークオブジェクトは関数呼び出しでどのような計算をするかを定義します。
例えば，xという入力をネットワークを渡し，結果としてyを受け取るコードは次のように書きます。

```
mlp = MLP(200, 10)
x = Variable(np.empty(10, 200))
y = mlp(x)
```

ネットワーク内で登録したパラメータは属性として呼び出すことができます。
例えば，self.l1というのは初期化で定義したl1を使うということを意味しています。

このように作ったMLPを分類器として使う場合には以下のようにL.Classifierを使ってモデルを作ります。
Classifierはデフォルトでは分類器softmax，学習時の損失関数はsoftmaxクロスエントロピー損失を使います。

```
model = L.Classifier(MLP(784, 10))
```

メモ
Chainerの特徴は"Define by Run"，つまり順に実行しながらネットワークを定義していきます。
この例の中では関数呼び出し__call__の中でネットワークを順次作っていますが，他の場所で作っても構いません。

## 課題

Chainで継承されたオブジェクトに登録されているパラメータ付関数はnamed_linksで呼び出すことができます。
例えば上の例の場合l1, l2が呼び出されます。
上の例を4層のニューラルネットワークに変更し，そのnamed_linksを表示してください。



