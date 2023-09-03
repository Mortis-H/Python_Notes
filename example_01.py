import numpy as np
import multiprocessing as mp

def hash_unique(arr):
    hash_table = {}
    unique_arr = []
    for a in arr:
        h = hash(a.data.tobytes())
        if h not in hash_table:
            hash_table[h] = True
            unique_arr.append(a)
    return unique_arr

# 創建影像
image = np.random.randint(2, size=(20000, 20000, 1))

# 切割影像
width, height, channel = image.shape
cut_size = 5
cut_images = []
for w in range(0, width, cut_size):
    for h in range(0, height, cut_size):
        cut_images.append(image[w:w+cut_size, h:h+cut_size, :])

# 將切割後的影像轉換為一維向量
flatten_images = [cut_image.flatten() for cut_image in cut_images]


# 刪除重複元素
unique_images = hash_unique(flatten_images)
unique_images_2D  = hash_unique(cut_images)
# 將一維向量轉換回矩陣
unique_cut_images = [unique_image.reshape(cut_size, cut_size, channel) for unique_image in unique_images]

# 結果
print("原始影像大小：", image.shape)
print("切割影像數量：", len(cut_images))
print("Unique影像數量：", len(unique_cut_images))
print("Unique 2D 影像數量：", len(unique_images_2D))
del image, cut_images, flatten_images
