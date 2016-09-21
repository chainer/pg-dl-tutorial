# Chainerの基本：Link

大部分の機械学習，少なくともニューラルネットワークはパラメータで特徴づけられた関数を用意し，このパラメータを調整することで学習を実現します。

Chainerではこのようなパラメータ付きの関数をLinkとよびます。

```
from chainer import links as L
```

ニューラルネットワークで最も広く利用される関数が[Linear](http://docs.chainer.org/en/stable/reference/links.html)であり，これは総結合層，またはアフィン変換とよばれ，次の関数で表現されます。

```
f(x; W, b) = Wx + b
```

関数において，;より後ろ側にあるのはそれがパラメータだということを意味し，それによって決めることを意味します。
Linearは初期化パラメータとして（入力の次元数, 出力の次元数）をうけとり，例えば3次元のベクトルから2次元へのベクトルへの変換は（3行2列の行列Wと2列のベクトルbからなる）

```
f = L.Linear(3, 2)
```

のように定義されます。

Linkのパラメータは属性に保存されており，それらはVariableです。
例えば，Linkの場合，Wとbのパラメータがあります。

```
f.W.data
f.b.data
```

Chainerの場合，Wのデフォルト値はガウシアン分布に従うランダムであり，bは0に初期化されます。これらの初期値はオプション引数で選ぶことができます。
また，デフォルトではbiasがありますが，nobias=Trueを指定することでbiasが無いLinearを作ることができます。

Linksでは次のようなメソッドがあります。

* add_param(name, shape)
　save, load時の対象となり，optimizationの対象となる
* add_persistent(name, value)
　save, load時の対象となる。
* addgrads(link)
　linkのgradient値を加算する。例えば分散学習時に使われる。
* children
* cleargrads()
　gradの値を0に初期化する。backward命令の前に呼び出す必要がある。
* copy()
　対象のlinkの子全てをコピーする。浅いコピーであり，パラメータのVariableはオブジェクトはコピーだが，それらのdataとgradient配列は共有される。linkの名前は初期化される。
* copyparams(link)
　linkからparameterをコピーする。
* namedlinks()
　全てのpath, linkを返す
* namedparams
　全てのpath, paramを返す
* serialize（serializer）
　このlinkオブジェクトをserializeする
* to_cpu()
　パラメータとpersistent値をCPUにコピーする
* to_gpu(device=None)
　パラメータとpersistnt値をGPUにコピーする
* xp
　今CPUとGPUのどちらにいるかにしたがって，CPUであればnumpy，GPUであればcupyを返す

# 課題

入力次元数が2の出力次元数が3のLinearを定義し，その重みの初期値を[[1, 2, 3], [4, 5, 6]]とした上で、メソッドのいくつかを試し，その結果を確かめよ