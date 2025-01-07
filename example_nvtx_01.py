import cupy as cp
import numpy as np
import nvtx
import time

# 設定資料大小
size = 10_000_000

# 建立 nvtx range
@nvtx.annotate("資料準備", color="green") 
def prepare_data():
    return np.random.rand(size)

# 主要運算函式
def main():
    nvtx.mark("開始運算")
    
    # 準備 NumPy 資料
    with nvtx.annotate("NumPy 資料產生", color="blue"):
        np_data = prepare_data()
    
    # 轉換到 GPU
    with nvtx.annotate("資料轉移到 GPU", color="yellow"):
        gpu_data = cp.array(np_data)
    
    # GPU 運算
    @nvtx.annotate("GPU 運算", color="red")
    def gpu_compute(data):
        return cp.sin(data) + cp.cos(data) * cp.exp(data)
    
    result_gpu = gpu_compute(gpu_data)
    
    # 資料轉回 CPU
    with nvtx.annotate("資料轉回 CPU", color="purple"):
        result_cpu = result_gpu.get()
    
    nvtx.mark("運算完成")
    
    return result_cpu

# 執行多次測試
for i in range(3):
    with nvtx.annotate(f"第 {i+1} 次迭代"):
        result = main()
        nvtx.mark(f"第 {i+1} 次完成")
