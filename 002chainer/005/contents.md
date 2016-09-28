# Chainerの基本：Link

ニューラルネットワークはパラメータを持った関数のパラメータを調整することで学習を実現します。

Chainerではこのような学習可能なパラメータを持った関数をLinkとよびます。

```
from chainer import links as L
```

例として，ニューラルネットワークで最も広く利用される関数である[Linear](http://docs.chainer.org/en/stable/reference/links.html)を紹介しましょう。
このLinearは総結合層，またはアフィン変換とよばれ，次の関数で表現されます。

```math
f(x; W, b) = Wx + b
```

関数において，$;$より後ろ側の変数はそれがパラメータだということを意味します。
Linearは初期化パラメータとして（入力の次元数, 出力の次元数）をうけとります　　
例えば入力が3次元のベクトルで出力が2次元のベクトルの場合，

```
f = L.Linear(3, 2)
```

のように定義されます。
これは内部では3行2列の行列Wと2列のベクトルbからなります。

Linkのパラメータは属性に保存されており，それらはVariableです。
例えば，Linkの場合，Wとbの属性があります。

```
f.W
f.b
```

Chainerの場合，Wのデフォルト値はガウシアン分布に従う乱数で初期化され，bは0に初期化されます。
これらの初期値はオプション引数で選ぶことができます。
また，デフォルトではバイアス項であるbがありますが，nobias=Trueを指定することでバイアスが無いLinearを作ることができます。

LinkはFunctionと同様に関数として呼び出すことができます。

多くのFunctionやLinkは入力として最初の次元数がバッチであるようなミニバッチ入力をうけとるように設計されています。
例えば，先程のLinearはバッチサイズがNの時，shapeが(N, 3)であるVariableを入力とし，shapeが(N, 2)でVariableを出力します。

```
x = Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32))
y = f(x)
print(y.data)
```

Linksでは次のようなメソッドがあります。
全て覚える必要はありませんが

* add_param(name, shape)
　新しくparameterを追加する。これはsave, load時の対象となり，optimizationの対象となる
* add_persistent(name, value)
　save, load時の対象となるパラメータを追加する。
* addgrads(link)
　linkのgradient値を加算する。例えば分散学習時に使われる。
* children
　子のlinkのgeneratorを返す。
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

## メモ

殆どのLinkには，それと同じ名前のFunctionが存在します。
例えばLinearもlinks.Linearとfunctions.Linearが存在します。
前者が学習可能パラメータを属性として持ったLinkであり，後者は学習可能パラメータを引数として受け取って計算Functionです。
links.Linearは内部でfunctions.Linearを呼び出して使っています。
もしユーザーが自分で学習対象パラメータを管理した上で同じ関数を使いたい場合はfunctions上で定義されている関数を直接呼び出して使うことができます。

## 課題

入力，出力がともに(N, 3, 4)であるような[bias](http://docs.chainer.org/en/stable/reference/links.html?highlight=link#bias)を作り，それをshapeが(2, 3, 4)であるVariableに適用し，その結果を表示せよ
http://docs.chainer.org/en/stable/reference/links.html?highlight=link#bias
