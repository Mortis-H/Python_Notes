# Python GPU Programming 教材

這是一份專為希望學習在 GPU 上進行高速運算的 Python 程式設計者所撰寫的教材。目標是幫助有多年 CPU 序列式程式設計經驗的工程師，在高效能運算（HPC）時代重新建構認知。

## 筆記內容

- 引導讀者從傳統的 CPU 思維轉換到 GPU 編程設計思維。
- 從簡單的範例出發，逐步深入 GPU 程式設計的核心概念。
- 探討性能瓶頸分析工具（如 Nsys）的使用，讓程式效能可觀測、可優化。

## 教材內容

1. **序章與環境建置**
   - 安裝與設定 CuPy
   - 建置 Windows 和 macOS 的 GPU 開發環境
2. **Python 與 GPU 的基礎**
   - 介紹 CuPy 與 cuDF
   - 初步了解 GPU 的執行架構（Thread, Block, Grid）
3. **效能分析工具：NVIDIA Nsight Systems**
   - 功能簡介
   - 使用範例
   - 結果解讀
4. **案例學習**
   - 將 CPU 程式搬移到 GPU 的實戰過程
   - 資料傳輸（D2H, H2D）與效能考量
   - 部分移植與全程 GPU 化的比較
5. **深入 CUDA 核心**
   - 使用 `cp.fuse` 優化運算
   - 使用 `cp.RawKernel` 寫自訂 CUDA 核心
