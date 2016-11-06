# 機械学習

ニューラルネットワークを含む多くの機械学習における学習タスクは最適なパラメータを探す問題です。
最適なパラメータは目的関数の最小化（最大化）問題を解くことで自動的に得られます。

一般に何かを学習させたいという場合は次のステップからなります。

1. 学習対象のモデルを定義する
2. 目的関数を定義する
3. 目的関数を最適化することで，モデルを学習する

これらを順番にみていきましょう。

## 1. 学習対象のモデルを定義する

パラメトリックモデルはパラメータθで特徴付けられた関数 $y=F(x;\theta)$ で表すことができます。
この関数 $F(x;\theta)$ は $x$ を受け取り， $y$ を返すような関数です。

この関数の挙動がパラメータ $\theta$ で変わることを示すために，引数とは違って $;\theta$ と表します。

Chainerではこのようなパラメータ $\theta$ で特徴付けられた関数はLinkとよばれます。
例えば，線形関数，またはアフィン変換を表すLinkであるLinearは

$f(x;\theta)=Wx+b$

という関数に対応します．この，パラメータ $\theta$ は行列 $W$ と，ベクトル $b$ であり， $\theta=(W,b)$ です。

例えば，今回のMNISTの場合，入力ベクトルxから0〜9の数を推定するようなモデルを作りたいと考えます。
モデルは一つ以上の学習可能な関数を組み合わせて構築

これはChainerでは次のように書けます。

```
class MLP(chainer.Chain): # MultiLayer Perceptron

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

順番にみていきましょう。Linkとは学習可能なパラメータを持つ関数であり，Linearは
上記の線形変換 $Wx+b$ の処理に対応します。
Linearの第一引数は，入力を何次元か，第二引数は出力を何次元かです。

Linearは学習可能な線形変換を表します。
作成したLinearは()を使って適用することが可能です。
`y = l(x)` は，線形変換を適用しそれを `y` に格納することを意味します。

## 2. 目的関数 $L(F(\theta))$ を定義する

次に学習によって達成したいことを表す目的関数を定義します。
例えば，回帰問題の場合は，予測した値 $y$ と実際の値 $t% が一致している場合は $0$ ，一致していない場合は大きな正の値をとるように
二乗誤差 $(y-t)^2$ を使います。

Chainerには，二乗誤差を表す `mean_squared_error` が用意されていますので，それを使います。

```
t = np.asarray([1, 8], dtype=np.float32)
t = Variable(t)
loss = F.mean_squared_error(y, t)
```

## 3. 目的関数を最小化するような $\theta$ を最適化問題を解くことで得る

最後に，目的関数を最小化するような $\theta$ を求めます。

```
from chainer import optimizer

opt = optimzier.SGD()
opt.setup(l)
opt.use_cleargrads()

...

loss.backward()
model.update()
```

Chainerの場合，Optimizerを使って学習させることに対応します。

これらを順に紹介していきます。
