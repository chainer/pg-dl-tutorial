# Chainerの基本：Variable

それではこの章からChainerの基本についてみていきます。

ChainerはDefine-by-runとよばれる思想に基づいて作られています。
これは計算手順を書くとそれ自身がネットワークの定義になるというものです。

計算手順を書く際には対象となる変数はVariableとよばれるオブジェクトを定義する必要があります。

はじめに，5という値一つからなるndarrayを作り，それを元にVariableを作ります。

```
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
```

このVariableオブジェクトは普通の数と同じように基本的な算術演算をすることができます。

例えば，y=x^2 - 2x + 1は

```
y = x**2 - 2 * x + 1
```

Variableオブジェクトを使った演算結果はVariableオブジェクトとなります。このyもVariableオブジェクトになっています。

Variableオブジェクトの値はdata属性で参照することができます。

```
y.data
```

# 課題

i=0...100について，Variableオブジェクトをxi=2*i+1とした上で，これらの二乗和であるVariableオブジェクトy=\sum_i xi^2を計算しその値を表示せよ