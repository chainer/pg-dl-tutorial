# Chainerの基本：Variable

それではこの章からChainerの基本についてみていきます。

ChainerはDefine-by-runとよばれる思想に基づいて作られています。
これは計算手順を書くとそれ自体がネットワークの定義になるというものです。

## メモ

ネットワークの定義に利用する変数はVariableとよばれるオブジェクトとして定義する必要があります。
Variableオブジェクトとして定義すると，以降このVariableオブジェクトを含む計算手順は全て追跡され計算グラフが自動的に作られます。

例として，5という値一つからなるndarrayを作り，それを元にVariableを作ってみましょう。

```
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
```

このVariableオブジェクトは普通の数と同じように基本的な算術演算をすることができます。
例えば，次のように実行できます。

```
y = x**2 - 2 * x + 1
```

Variableオブジェクトを使った演算結果はVariableオブジェクトとなります。
この例の場合，yもVariableオブジェクトです。

Variableオブジェクトの値はdata属性で参照することができます。

```
y.data
```

また，numpyと同じ添字アクセスを備えています。

```
z_data = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
z = Variable(z_data)
print(z[:,1].data) # [3, 6]
```

# 課題

i=0...100について，Variableオブジェクトをxi=2*i+1とした上で，これらの二乗和であるVariableオブジェクト$y=\sum_i xi^2$を計算しその値を表示せよ
