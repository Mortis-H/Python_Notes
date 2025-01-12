import cupy as cp
import numpy as np

# 建立測試資料
array_shape = (3, 400000)
data = cp.random.random(array_shape)  # 建立隨機測試資料

# 建立一個 boolean mask (shape = 400000)
mask = cp.random.random(400000) > 0.5  # 隨機產生 boolean mask

# 方法1: 使用迴圈
filtered_array1 = cp.zeros_like(data)
for i in range(3):
    filtered_array1[i,:][mask] = data[i,:][mask]

# 方法2: 使用 take
mask2 = cp.expand_dims(mask, axis=0)
filtered_array2 = cp.where(cp.tile(mask2, (3, 1)), data, 0)

# 比較兩個結果
print("比較兩種方法的結果:")
cp.testing.assert_array_equal(filtered_array1, filtered_array2)
print("兩種方法結果完全相同!")