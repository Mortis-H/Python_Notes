import cupy as cp
import numpy as np

# 建立測試資料
size = 10
cpu_array = np.zeros(size)
gpu_array = cp.zeros(size)

# 重複的索引
indices = np.array([3, 3, 3, 5, 5])
values = np.array([1, 2, 3, 4, 5])

print("原始陣列:")
print("CPU:", cpu_array)
print("GPU:", gpu_array)

print("\n使用重複索引賦值:")
# CPU 上的行為 - 會使用最後一個值
cpu_array[indices] = values
print("CPU 結果:", cpu_array)

# GPU 上的行為 - 結果不確定
gpu_array[indices] = values
print("GPU 結果:", gpu_array)

# 再執行一次 GPU 賦值看看結果是否不同
gpu_array = cp.zeros(size)
gpu_array[indices] = values
print("GPU 第二次結果:", gpu_array)

