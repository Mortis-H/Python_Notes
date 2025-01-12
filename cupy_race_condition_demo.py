# 建立一個簡單的費波那契數列計算
# 方法1: 使用動態規劃(迭代)避免遞迴
import cupy as cp
n = 10
fib = cp.zeros(n, dtype=cp.int64)  # 使用 GPU 陣列
fib[0] = 0
fib[1] = 1

print("開始計算費波那契數列(迭代法 + GPU):")
for i in range(2, n):
    fib[i] = fib[i-1] + fib[i-2]
    print(f"第 {i} 個數字是: {fib[i].get()}")  # 使用 .get() 取回 CPU

print("\n最終費波那契數列:")
print(fib.get())  # 使用 .get() 取回 CPU

# 方法2: 使用矩陣快速冪方法 + GPU
def matrix_fib(n):
    if n <= 0:
        return 0
    base = cp.array([[1, 1], [1, 0]], dtype=cp.int64)  # 使用 GPU 陣列
    result = cp.array([[1, 0], [0, 1]], dtype=cp.int64)
    n = n - 1
    
    while n > 0:
        if n % 2 == 1:
            result = cp.dot(result, base)  # 使用 GPU 矩陣乘法
        base = cp.dot(base, base)
        n = n // 2
        
    return result[0][0].get()  # 使用 .get() 取回 CPU

print("\n使用矩陣快速冪方法(GPU加速):")
for i in range(n):
    print(f"第 {i} 個數字是: {matrix_fib(i)}")

# 方法3: 使用通項公式(黃金比例)
# 注意: 這個方法主要是數學運算，使用 GPU 加速效果有限
def formula_fib(n):
    phi = (1 + 5 ** 0.5) / 2
    return int((phi ** n - (-phi) ** -n) / 5 ** 0.5)

print("\n使用通項公式:")
for i in range(n):
    print(f"第 {i} 個數字是: {formula_fib(i)}")
