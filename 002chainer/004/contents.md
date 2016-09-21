# Chainerの基本：Funciton

Functionオブジェクトは学習可能なパラメータを持たない関数です。但し通常の関数とは違って逆誤差伝播が計算できるように，前向き計算に加えて，後ろ向き計算ができるようになっています。

また，CPUとGPUの両方における計算が定義されています。

Functionを利用するには，そのcallオブジェクトを呼び出すだけです。
例えば，relu(x) = max(x, 0)で定義されるrelu関数は次のようによびだします。

```
from chainer import functions as F

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
y = F.relu(x)
```

これらの関数は要素毎であったり，関数の形を変えたりします。

なお，Variableの時に普通の算術演算ができるという話しでしたが、それは実際には算術演算を対応するFunction呼び出しをオーバーロードして実現されています。

例えば，

```
x = Variable(x_data)
y = Variable(y_data)
z = x + y
```

は，

```
z = F.Add(x, y)
```

を呼び出しています。

Functionは計算履歴を追跡する仕掛けが含まれています。

Functionを適用した結果はVariableなので，backwardを呼び出して勾配を求めることができます

```
z.backward()

## メモ

後ろ向き計算とは、 逆誤差伝播法で使われる出力の入力についての勾配，つまりy = f(x)の時 \partial y / \partial xの計算です。例えばy=3x^2の場合は後ろ向き計算はyのxについての微分，つまり6xが後ろ向きの結果になります。

## 課題

x=[3, 4, 5]の時にこれをVariableにした上で，exp(x)+sin(x)の勾配を求めよ

