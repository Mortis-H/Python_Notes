import cupy as cp
import numpy as np
import nvtx

# 建立測試資料
array_shape = (3, 400000)
data = cp.random.random(array_shape)  # 建立隨機測試資料

# 建立一個 boolean mask (shape = 400000)
mask = cp.random.random(400000) > 0.5  # 隨機產生 boolean mask

for _ in range(5):

    # 方法1: 使用迴圈
    with nvtx.annotate("method1"):
    filtered_array1 = cp.zeros_like(data)
        for i in range(3):
            filtered_array1[i,:][mask] = data[i,:][mask]

    # 方法2: 使用 take
    with nvtx.annotate("method2"):
        mask2 = cp.expand_dims(mask, axis=0)
        filtered_array2 = cp.where(cp.tile(mask2, (3, 1)), data, 0)