import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent  # 使用 scipy 内置的 ascent 图像
from skimage.transform import resize

# ==========================================
# 1. 加载数据及预处理
# ==========================================
print("=== 1. 加载与预处理图像 ===")
# 1.1 加载经典图像 ascent (通常是一个 512x512 的灰度图)
img_raw = ascent()

# 1.2 预处理: resize 至 (128, 128) 以加速计算
# skimage 的 resize 默认会将像素值归一化到 [0, 1] 区间，且转为 float
img_resized = resize(img_raw, (128, 128), anti_aliasing=True)
img = img_resized.astype(np.float32)

print(f"原始图像 shape: {img_raw.shape}")
print(f"预处理后图像 shape: {img.shape}, 数据类型: {img.dtype}, 值范围: [{img.min():.2f}, {img.max():.2f}]")


# ==========================================
# 2. 实现 2D 卷积函数
# ==========================================
def my_conv2d(img, kernel, stride=1, padding=0):
    """
    手写 2D 卷积函数 (无 batch 维度，单通道)
    :param img: 输入图像, 形状 (H, W)
    :param kernel: 卷积核, 形状 (K, K)
    :param stride: 步长 (S)
    :param padding: 零填充大小 (P)
    :return: 卷积后的特征图, 形状 (H_out, W_out)
    """
    H, W = img.shape
    K = kernel.shape[0]
    assert kernel.shape[0] == kernel.shape[1], "卷积核必须是方阵"

    # 2.1 公式计算输出尺寸并 Assert
    # H_out = floor((H + 2P - K) / S) + 1
    # W_out = floor((W + 2P - K) / S) + 1
    H_out = int(np.floor((H + 2 * padding - K) / stride)) + 1
    W_out = int(np.floor((W + 2 * padding - K) / stride)) + 1

    # 2.2 对原图进行零填充 (Zero Padding)
    if padding > 0:
        # np.pad 的 pad_width 格式为 ((top, bottom), (left, right))
        padded_img = np.pad(img, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded_img = img.copy()

    # 初始化输出特征图
    out = np.zeros((H_out, W_out), dtype=np.float32)

    # 2.3 双重循环实现滑动窗口卷积
    for i in range(H_out):
        for j in range(W_out):
            # 计算当前窗口在 padded_img 中的切片边界
            h_start = i * stride
            h_end = h_start + K
            w_start = j * stride
            w_end = w_start + K

            # 提取当前区域
            region = padded_img[h_start:h_end, w_start:w_end]

            # 逐元素相乘并求和 (Element-wise multiplication and sum)
            out[i, j] = np.sum(region * kernel)

    return out


# ==========================================
# 3. 边缘检测实验 (Sobel 核)
# ==========================================
print("\n=== 2. 应用 Sobel 核进行边缘检测 ===")

# 3.1 定义 Sobel X 卷积核 (用于提取垂直边缘)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# 3.2 定义 Sobel Y 卷积核 (用于提取水平边缘)
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# 执行卷积操作 (使用 padding=1 保持原图尺寸不变，stride=1)
edge_x = my_conv2d(img, sobel_x, stride=1, padding=1)
edge_y = my_conv2d(img, sobel_y, stride=1, padding=1)

print(f"Sobel X 输出 shape: {edge_x.shape}")
print(f"Sobel Y 输出 shape: {edge_y.shape}")

# ==========================================
# 4. 实现 Max Pooling (最大池化) 函数
# ==========================================
print("\n=== 3. 执行 Max Pooling ===")


def my_maxpool2d(img, kernel_size=2, stride=2):
    """
    手写 2D 最大池化函数
    :param img: 输入图像, 形状 (H, W)
    :param kernel_size: 池化窗口大小 (K)
    :param stride: 步长 (S)
    :return: 下采样后的图像, 形状 (H_out, W_out)
    """
    H, W = img.shape
    K = kernel_size
    S = stride

    # 公式计算输出尺寸 (默认无 padding)
    H_out = int(np.floor((H - K) / S)) + 1
    W_out = int(np.floor((W - K) / S)) + 1

    out = np.zeros((H_out, W_out), dtype=np.float32)

    # 循环实现滑动窗口求最大值
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * S
            h_end = h_start + K
            w_start = j * S
            w_end = w_start + K

            region = img[h_start:h_end, w_start:w_end]
            # 求当前窗口内的最大值
            out[i, j] = np.max(region)

    return out


# 对原图进行 Max Pooling
pooled_img = my_maxpool2d(img, kernel_size=2, stride=2)
print(f"Max Pooling 输出 shape: {pooled_img.shape} (预期为 64x64)")

# ==========================================
# 5. 可视化结果
# ==========================================
plt.figure(figsize=(15, 10))

# 1. 原图
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original Image\n{img.shape}')
plt.axis('off')

# 2. 垂直边缘 (Sobel X)
plt.subplot(2, 3, 2)
# 注意：卷积输出可能包含负数，绘图时 cmap='gray' 会自动拉伸显示
plt.imshow(edge_x, cmap='gray')
plt.title(f'Vertical Edges (Sobel X)\n{edge_x.shape}')
plt.axis('off')

# 3. 水平边缘 (Sobel Y)
plt.subplot(2, 3, 3)
plt.imshow(edge_y, cmap='gray')
plt.title(f'Horizontal Edges (Sobel Y)\n{edge_y.shape}')
plt.axis('off')

# 4. Max Pooling 结果
plt.subplot(2, 3, 4)
plt.imshow(pooled_img, cmap='gray')
plt.title(f'Max Pooling (2x2, S=2)\n{pooled_img.shape}')
plt.axis('off')

plt.tight_layout()
plt.savefig('lab5_conv_visual.png')
plt.show()

print("\n已保存可视化结果为 'lab5_conv_visual.png'")