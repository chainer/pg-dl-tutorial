# -*- coding: utf-8 -*-
import numpy as np
x = np.empty((3, 640, 480), dtype=np.int32)

print(x)  # xをランダムに初期化した時の値

print(x.shape)  # xの寸法

print(x.ndim)  # xの軸数

print(x.size)  # xの全要素数

print(x.dtype)  # xの型

# x[i, i, i]=1, それ以外は0となるようなx
