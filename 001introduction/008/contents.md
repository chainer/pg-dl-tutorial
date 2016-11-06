# NumPy, Cupy, テンソル

Chainerの中で主要なオブジェクトはNumPyで定義される `ndarray` よばれる多次元配列です。
これは数学的にはテンソルともよばれます。

NumPyの使い方については素晴らしい資料やチュートリアルが存在します。例えば
[チュートリアル](http://naoyat.hatenablog.jp/entry/2011/12/29/021414)
を参照してください。

ここでは，Chainerを扱う上で最低限必要なNumPyの `ndarray` についての知識について説明します。

NumPyは慣習としてnpとして利用します。

```
import numpy as np
```

NumPyの主要なオブジェクトは `ndarra` yであり，多次元配列を表します。
`ndarray` は次元または軸（axis）を持ち，軸の数をrankとよびます。

多次元配列はそれぞれ `rank==0` の時はスカラー， `rank==1` の時はベクトル， `rank==2` の時は行列， `rank>=3` の時はテンソルとよばれます。
`ndarray` の寸法（shape）は各軸の配列長を表す整数からなるタプル（例 (3,) (2, 3, 4)）で表され， `shape` 属性として取得できます。
`ndarray` の値は全て同じ型を持ち， `dtype` 属性で参照できます。
ディープラーニングで扱う場合， `dtype` は殆どの場合 `np.float32` , `np.int32` です。

それでは長さ3のベクトルvと2行4列からなる行列mを作ってみましょう。
最初の引数がshapeを指定し，二つ目の引数が型を指定します。

```
v = np.zeros((3,), np.float32)
m = np.zeros((2, 4), np.float32)

print(m.shape)
print(m.dtype)
```

格納されている値は添字を使って参照したり，また添字を使って代入することができます。

```
print(m[0, 2])
m[0, 2] = 7
print(m[0, 2])
```

例えば，スカラーは0次元配列，ベクトルは1次元配列，行列は2次元配列です。

ChainerではNumPyと同じコードでGPU上での演算を実現するcupyとよばれるライブラリを使います。
それによりCPUかGPUかを区別せずに同じコードを書くことができます。

```
is_gpu = True # CPUの場合はFalse

xp = cupy if is_gpu else numpy

m1 = xp.zeros((3, 4))
```

## 課題

shapeが `(10, 5, 4)` であり， `x[i, i, i] = 1` ，つまり1軸目と2軸目と3軸目の添字が一致する場合のみ1になり，それ以外は全て0になるよな `ndarray` を作り，それを表示せよ。
