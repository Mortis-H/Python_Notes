import cupy as cp
import time
import nvtx

# 建立大型測試資料
size_1  = 1_000_000
size_2 = 5_00
iterations = 3

def test_cupy():
    # 建立 Row-Major 和 Column-Major 陣列
    with nvtx.annotate("建立陣列"):
        with nvtx.annotate("Row-Major"):
            row_major = cp.zeros((size_1, size_2),order='F')  # Row-Major (C-order)
        with nvtx.annotate("Column-Major"):
            col_major = cp.zeros((size_2, size_1), order='F')  # Column-Major (F-order)

    # 測試按列存取 (Column-wise access)
    print("按存取測試:")

    # Column-Major 按列存取
    with nvtx.annotate("Column-Major 存取"):
        for _ in range(iterations):
            for i in range(size_2):
                col_major[i,:] += 1  # 按列存取 Column-Major 陣列

    # Row-Major 按列存取
    with nvtx.annotate("Row-Major 存取"):
        for _ in range(iterations):
            for i in range(size_2):
                row_major[:,i] += 1  # 按列存取 Row-Major 陣列


for _ in range(3): 
    test_cupy()
