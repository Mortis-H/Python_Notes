# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
import time
import nvtx

# 導入作業系統相關功能
import os

# 設定 CuPy 的快取目錄位置
os.environ['CUPY_CACHE_DIR'] = r"D:/cupy_demo/cupy_cache"

# rmm
import rmm
from rmm.alocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)

# 設定 GPU 記憶體分配器
# 建立一個 4GB 大小的記憶體池來管理 GPU 記憶體
cp.cuda.set_allocator(cp.cuda.MemoryPool(4 * 1024 * 1024 * 1024).malloc)

# 設定 GPU 計算的精確度
cp.cuda.set_default_dtype(cp.float32)  # 設定為單精度浮點數以提升效能

# 啟用自動記憶體管理
cp.cuda.set_pinned_memory_allocator()

# 設定 GPU 設備
device = cp.cuda.Device(0)  # 使用第一個 GPU
device.use()

# 顯示 GPU 資訊
print(f"使用的 GPU: {cp.cuda.runtime.getDeviceName()}")
print(f"可用記憶體: {cp.cuda.runtime.memGetInfo()[0]/1024**3:.2f} GB")

# 設定 kernel 編譯選項
cp.cuda.compiler.compile_with_cache = True  # 啟用 kernel 快取

@cp.fuse()
def complex_calc(x):
    return (cp.sin(x) ** 2 + 
            cp.cos(x) ** 3 + 
            cp.exp(-x) * cp.log(x + 1))

def test_cupy():
# 設置數據大小
    rows, cols = 10_000_000, 12

    # Step 1: 建立 Numpy 陣列 (Row-Major 和 Column-Major)
    with nvtx.annotate("建立 Numpy 陣列"):
        row_major_array = np.asarray(np.random.rand(rows, cols))  # 強制指定為 Row-Major (C-order)
        col_major_array = np.asarray(np.random.rand(cols, rows))  # 強制指定為 Column-Major (F-order)

    # Step 2: 將 Numpy 陣列傳輸到 CuPy
    # 測試 Column-Major 傳輸到 GPU
    with nvtx.annotate("Column-Major 傳輸到 GPU"):
        cp_col_major = cp.array(col_major_array)

    # 測試 Row-Major 傳輸到 GPU
    with nvtx.annotate("Row-Major 傳輸到 GPU"):
        cp_row_major = cp.array(row_major_array)


    # Step 3: 測試在 GPU 上的複雜運算
    # 定義 fused kernel 函數

    # Row-Major 運算
    with nvtx.annotate("Row-Major GPU 運算"):
        # 使用 fused kernel 進行複雜運算
        row_major_result = complex_calc(cp_row_major)

    # Row-Major 運算 (不使用 fuse)
    with nvtx.annotate("Row-Major GPU 運算 (無 fuse)"):
        # 進行多個複雜運算: sin(x)^2 + cos(x)^3 + exp(-x) * log(x+1)
        row_major_result_no_fuse = (cp.sin(cp_row_major) ** 2 + 
                                  cp.cos(cp_row_major) ** 3 + 
                                  cp.exp(-cp_row_major) * cp.log(cp_row_major + 1))

    # Column-Major 運算
    with nvtx.annotate("Column-Major GPU 運算"):
        # 使用 fused kernel 進行複雜運算
        col_major_result = complex_calc(cp_col_major)

    # Column-Major 運算 (不使用 fuse)
    with nvtx.annotate("Column-Major GPU 運算 (無 fuse)"):
        # 相同的複雜運算
        col_major_result_no_fuse = (cp.sin(cp_col_major) ** 2 + 
                                  cp.cos(cp_col_major) ** 3 + 
                                  cp.exp(-cp_col_major) * cp.log(cp_col_major + 1))

    # Step 4: 將結果傳回 CPU
    # Column-Major 回傳
    with nvtx.annotate("Column-Major 傳回 CPU"):
        col_major_result_host = col_major_result.get()

    # Row-Major 回傳
    with nvtx.annotate("Row-Major 傳回 CPU"):
        row_major_result_host = row_major_result.get()


for _ in range(5):
    test_cupy()
