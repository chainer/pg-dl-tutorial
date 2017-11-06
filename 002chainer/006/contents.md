# Chainerの基本：Chain

多くのニューラルネットワークは複数のLinkから構成されます。
例えば，多層パーセプトロンは複数のLinear層からなります。

Chainerでは複数のLinkをまとめて一つのオブジェクトChainとして扱うことができます。
Chainはユーザーがネットワークを定義する際に利用されます。

```
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
```

Chainを継承すると，その中に含まれる複数のLinkの管理やCPU/GPU間のデータ移動などが実現されます。

ChainではLinkを登録するには，例のように`with self.init_scope()`の中で登録します。

Chainの中に含まれるLinkを子Linkとよびます。
例えばさきほどの例ではl1とl2がMyChainの子Linkです。
なお，Chain自身もLinkを継承しています。
そのため，あるChainを他のChainの子リンクとして使うことができます。

Chainの子リンクは属性としてアクセスすることができます。

```
c = MyChain()
print(c.l1.W.data)
```

また，Chainでは各Linkを名前付きで定義していましたが，任意個のLinkのリストを受け取るChainListを使うこともできます。

```
class MyChainList(ChainList):
    def __init__(self):
        super(MyChain, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)
```

# 課題

正の整数$n$を初期化パラメータとして受け取り，$n$個のLinear(3, 3)を子Link(l1, l2, ..., ln)として含み，入力に対し$l1, l2, ..., ln$を順に適用するようなChainオブジェクトを作れ。
