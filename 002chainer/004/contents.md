# Chainerの基本：Function

Functionオブジェクトは学習可能なパラメータを持たない関数です。
但し通常の関数とは違って逆誤差伝播が計算できるように，前向き計算に加えて，後ろ向き計算ができるようになっています。

また，CPUとGPUの両方における計算が定義されています。
内部では，foward_cpu, foward_gpu, backward_cpu, backward_gpuの四種類の実装がされています。

Functionを利用するには，Functionのインスタンスを作成後に関数として呼びだします。
ニューラルネットワークで利用する多くの関数がchainer.functionsで既に実装されています。
例えば，$\mathrm{relu}(x) = \max(x, 0)$で定義されるrelu関数は次のように呼び出します。

```
from chainer import functions as F

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = F.relu(x)
```

これらの関数の多くは要素毎に対する関数適用であり，計算結果のshapeは計算元と変わりませんが，
一部の関数はshapeが変わります。


なお，Variableは基本的な算術演算ができるという話しでしたが、それらは実際には算術演算を対応するFunction呼び出しをオーバーロードして実現されています。

例えば，

```
x = Variable(x_data)
y = Variable(y_data)
z = x + y
```

は，内部ではそのFunctionオブジェクト

```
z = F.Add(x, y)
```

を呼び出しています。

Functionは計算履歴を追跡する仕掛けが含まれています。
Functionを適用した結果はVariableであり，backwardを呼び出して勾配を求めることができます

```
z.backward()
```

Variableが保持するデータはndarrayなので，numpyと同様の様々な配列変換（例えばreshapeやtransposeなど）を使いますがこの場合も計算結果が追跡できるようにfunctionsにある対応する関数を呼び出す必要があります。

```
z = Variable(np.array([[10, 20], [30, 40]], dtype=np.float32))
zz = F.transpose(z)
print(zz.data)
```


## メモ

後ろ向き計算とは、 逆誤差伝播法で使われる出力に対する入力についての勾配，つまり$y = f(x)$の時 $\partial y / \partial x$の計算です。例えば$y=3x^2$の場合は後ろ向き計算は$y$の$x$についての微分，つまり$6x$が後ろ向きの結果になります。
後ろ向き計算は効率的に計算でき，多くの場合前向き計算とほぼ同じ計算量で求めることができます。

## 課題

$x=[3, 4, 5]$の時，$\exp(x)+\sin(x)$の勾配を求めて表示せよ。
