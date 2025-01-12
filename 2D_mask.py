import cupy as cp
import numpy as np
import nvtx

# 建立測試資料
array_shape = (3, 400000)
data = cp.random.random(array_shape)  # 建立隨機測試資料

# 建立一個 boolean mask (shape = 400000)
mask = cp.random.random(400000) > 0.5  # 隨機產生 boolean mask
mask2 = cp.expand_dims(mask, axis=0)

for _ in range(5):

    # 方法1: 使用 boolean indexing
    with nvtx.annotate("method1"):
        # 使用 boolean indexing 直接對整個陣列操作,避免迴圈
        filtered_array1 = cp.zeros_like(data)
        filtered_array1[:, mask] = data[:, mask]

    # 方法2: 使用 take
    with nvtx.annotate("method2"):
        filtered_array2 = cp.where(cp.tile(mask2, (3, 1)), data, 0)
    
    # 方法3: 不用cp.tile
    with nvtx.annotate("method3"):
        filtered_array3 = cp.where(cp.asarray([mask2, mask2, mask2]), data, 0)
    
        # 比較三個方法的結果
        print("方法1和方法2結果是否相同:", cp.allclose(filtered_array1, filtered_array2))
        print("方法1和方法3結果是否相同:", cp.allclose(filtered_array1, filtered_array3))
        print("方法2和方法3結果是否相同:", cp.allclose(filtered_array2, filtered_array3))
