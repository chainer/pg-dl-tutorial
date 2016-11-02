# モデルの定義

それでは次に学習対象のモデルを定義します。

今回は3層からなるニューラルネットワークの例をあげます。



x ---> h1 ----> 
   l1 -> relu -> l2 -> relu -> l3 -> y

```
import chainer
from chainer import links as L
from chainer import functions as F

...

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units), # n_units -> n_units
            l3=L.Linear(None, n_out)  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

モデルを定義するオブジェクトはchainer.Chainを継承します。
Chainを継承することで，後でこのモデルを保存したり読み込んだりすることができます。

モデルを定義する際，初期化でネットワーク内で利用するパラメータ付き関数Linkを登録します。
ここではLinearであるl1とl2を登録しています。
Linearは線形変換であり，初期化引数として入力次元数と出力次元数をうけとります。
入力次元数について，それがNoneの時，実際にそれが呼び出された時，次元数を引数から推定してくれます。

ChainにおいてLinkの登録は例のように__init__の中で定義することもできますし，後で
add_link(name, link)
のように定義することもできます。

また，__call__の中でこのモデルはどのような関数かを定義します。
初期化時に登録されたLinkはself.l1のように属性として参照できます。

```
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

ここでは

x -> l1 -> relu -> l2 -> relu -> l3

のような変換を適用しています。
最後にreluを適用していないことに注意してください。

このMLPは，関数呼び出しをすることができます。

このネットワークオブジェクトは関数呼び出しでどのような計算をするかを定義します。
例えば，xという入力としてMLPを呼び出した場合は

```
mlp = MLP(200, 10)
x = Variable(np.empty(10, 200))
y = mlp(x)
```

のようになります。

最後に，

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



