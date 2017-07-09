
# 1. 学習対象のモデルを定義する (3)

Chainerでもう一つ重要なオブジェクトとしてFunctionがあります。
FuncitonはLinkとは違って，学習可能なパラメータを持ちません。
つまり，学習によって挙動を変えません。

ディープラーニングで利用されている代表的な関数は `chainer.functions` で定義されています。
また，自分で新しいFunctionを作ることもできます。

以降では，この  `chainer.functions` をFとして使えるようにします。

```
from chainer import functions as F
```

例えば，ディープラーニングでよく使われるReLUとよばれる非線形関数 $f_{relu}$ は

$$f_{relu}(x)=max(x,0)$$

で定義されます。
つまり，もし $x$ が $0$ よりも大きければ $x$ をそのまま返し，もし小さければ $0$ を返すような関数です。

Chainerでは次のように記述できます。

```
from chainer import functions as F
...
y2 = F.relu(x)
```

これらのLinkとFunctionを組みわせて複雑な関数を作ることができます。
例えば，前回の例のLinearを適用した後にReLUを適用した結果は次のように計算されます。

```
y3 = F.relu(lin(x))
```

## 課題

ReLUと並んで重要な非線形関数として，sigmoidがあります。
例をsigmoidに変えて，その結果を表示してください。

[関数一覧](http://docs.chainer.org/en/stable/reference/functions.html)
