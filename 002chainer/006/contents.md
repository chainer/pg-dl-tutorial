# Chainerの基本：Chain

多くのニューラルネットワークは複数のLinkから構成されます。
例えば，多層パーセプトロンは複数のLinear層からなります。

Chainerでは複数のLinkをまとめて一つのオブジェクトChainとして扱うことができます。

```
class MyChain(Chain):
  def __init__(self):
    super(MyChain, self).__init__(
      l1=L.Linear(4, 3),
      l2=L.Linear(3, 2),
    )

  def __call__(self, x):
    h = self.l1(x)
    return self.l2(h)
```

Chainを継承しているのは，Chainがそこに含まれる複数のLinkの管理やCPU/GPU間の移動などの機能をまかなってくれるからです。

ChainではLinkを登録するには，例のように初期化の中で登録するか，add_link(name, link)を使って登録

Chainの中に含まれるLinkを子Linkとよびます。例えば上の例ではl1とl2がMyChainの子リンク
なお，ChainはLinkを継承しています。そのため，MyChainを他のChainの子リンクとして使うことができます。

Chainでは各Linkを名前付きで定義していましたが，任意個のLinkのリストを受け取るChainListとよばれるものも存在します

# 課題

正の整数nを初期化パラメータとして受取，n個のLinear(3, 3)を子Link（l1, l2, ..., ln)として含み，入力に対しl1, l2, ..., lnを順に適用するようなChainオブジェクトを作れ
