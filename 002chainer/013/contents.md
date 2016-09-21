# Chainerの基本：MNIST 例

ようやく学習ができるようになりました。

modelによる入力xに対する予測結果をyとします。
そして，yとtとの間で測ったクロスエントロピー損失をlossとします。
そして，optを更新します。

この学習自体は何回も学習データを回す必要があるのでepoch_num回ループするようにします。
```
epoch_num = 5
for epoch in xrange(epoch_num):
    train_loss_sum = 0
    train_accuracy_sum = 0
    for i in xrange(0, train_num, batchsize):
        batch = train_iter.next()
        x = Variable(np.asarray([s[0] for s in batch]))
        t = Variable(np.asarray([s[1] for s in batch]))
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        opt.update()
        train_loss_sum += loss.data
        train_accuracy_sum += F.accuracy(y, t).data
```

次に評価用データセットで性能を評価するコードです。
この部分は学習と殆ど同じで唯一の違いはbackwardを呼ばず，modelの更新をしない部分です。
```
...
  test_loss_sum = 0
  test_accuracy_sum = 0
  for i in xrange(0, test_num, args.batchsize):
      batch = train_iter.next()
      x = Variable(xp.asarray([s[0] for s in batch]))
      t = Variable(xp.asarray([s[1] for s in batch]))
      y = model(x)
      loss = F.softmax_cross_entropy(y, t)
      test_loss_sum += loss.data.get()
      test_accuracy_sum += F.accuracy(y, t).data.get()
...
```

```

```

## 課題

実際に実行し，精度がどのように代わっていくのかを調べよ。



