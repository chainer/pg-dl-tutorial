# 3. 目的関数を最適化することで，モデルを学習する (4)

勾配情報に基づきパラメータを更新する手法が `chainer.optimizers` にサポートされています。

```
from chainer import optimizers
```

代表的な最適化手法はSGD, RMSprop, Adamなどです。

```
opt = optimizers.Adam()
opt.setup(model)
```

最適化エンジンがどの学習可能な関数を目標とするかは `setup` で設定します。
そして，勾配を求めて，その勾配情報を元に最適化します。

```
loss.backprop()
opt.update()
```

誤差逆伝搬法は強力で多くの関数の勾配を正確にかつ高速に求めることができます。
そのため，非常に多くのパラメータを持つモデルの場合でも効率的に学習することができます

