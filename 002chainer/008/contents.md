# Chainerの基本：Optimization

Chainerでは様々な最適化手法がサポートされています。

optimizerの使い方はoptimizerを用意し，次にoptimizerの最適化対象となるlinkをsetup()で設定します。

```
from chainer import optimizers

optimizer = optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(model)
```

この場合，modelというlinkが最適化対象になります。

代表的なoptimzierとして次の三つがあります。

* SGD
* Adam
* RMSProp

どれを使えばよいかはそれぞれ異なる最適化手法に基づいているので一概にいえませんが，安定して最適化できるのはAdam，精度が出やすいのはRMSPropという特徴があります。SGDは最も単純な学習速にもとづいており二つの手法に比べると性能は悪くデバッグ目的以外では使う必要はありません。

optimizerの使い方は三つあります。後の方がより使いやすくカスタマイズしにくい方法になります。

* ユーザーがbackward()などで勾配を求めて，引数なしのupdate()を呼び出す。この場合，cleargrads()を最初に呼ぶ必要がある

```
model.cleargrads()
loss.backward()
optimizer.update()
```

* 損失関数をupdate()に渡す。この場合，cleargrads()はupdate内で自動的によばれる

```
def lossfun(args...):
  ...
  return loss
optimizer.update(lossfun, args...)
```

* Trainerを利用する

これについては後の章で紹介します。

これらの使い分けですが，

学習問題が典型的な問題でありTrainerが既に用意されている，またはTrainerを書けるのであればTrainerを使うのが望ましいです。Trainerは学習ステップを抽象化し，切り替えることを可能とします。

何か特別な処理をしない通常の学習問題はupdateに損失関数を渡す二つ目の方法がcleargradsの呼び忘れがなくて簡潔に書けるので望ましいです。

最初の方法は何か通常の学習とは違う特別な処理をしたい場合（例えばgradientを参照したい，操作したいなど）に自分で勾配を設定できるので望ましいです。