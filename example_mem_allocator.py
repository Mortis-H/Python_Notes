import cupy as cp
import numpy as np
import nvtx
import time

# 設定記憶體池大小為 4GB
def setup_memory_pool():
    mempool = cp.cuda.MemoryPool(4 * 1024 * 1024 * 1024)
    cp.cuda.set_allocator(mempool.malloc)
    return mempool

print("測試不同記憶體分配策略的效能差異...")

# 不使用記憶體池的測試
with nvtx.annotate("Without Memory Pool"):
    # 重置為預設分配器
    cp.cuda.set_allocator()
    cp.get_default_memory_pool().free_all_blocks()
    
    start_time = time.time()
    for i in range(100):
        with nvtx.annotate(f"Allocation {i}"):
            # 重複分配和釋放記憶體
            x = cp.ones((1000, 1000), dtype=cp.float32)
            del x
    
    print("不使用記憶體池耗時:", time.time() - start_time, "秒")

# 使用記憶體池的測試
with nvtx.annotate("With Memory Pool"):
    mempool = setup_memory_pool()
    
    start_time = time.time()
    for i in range(100):
        with nvtx.annotate(f"Allocation {i}"):
            # 重複分配和釋放記憶體
            x = cp.ones((1000, 1000), dtype=cp.float32)
            del x
    
    print("使用記憶體池耗時:", time.time() - start_time, "秒")

# 清理記憶體
cp.get_default_memory_pool().free_all_blocks()

