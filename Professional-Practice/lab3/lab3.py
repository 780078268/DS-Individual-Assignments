import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 准备数据及预处理
X, y = load_digits(return_X_y=True)

# 划分：train/val/test = 60/20/20 (stratify=y 保证各类别比例一致)
# 先划出 20% 作为 test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 剩下的 80% 中，再划出 1/4 作为 val (0.8 * 0.25 = 0.2)，剩下作为 train (0.6)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 2. 网络参数初始化
D = 64  # 输入维度
H = 128  # 隐藏层维度
C = 10  # 输出维度 (10类)

# 权重初始化: W ~ N(0, 0.01), b = 0
np.random.seed(42)
W1 = np.random.randn(D, H) * 0.01
b1 = np.zeros(H)
W2 = np.random.randn(H, C) * 0.01
b2 = np.zeros(C)

print(f"参数形状打印:")
print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")


# 3. 核心激活函数与 Loss
def relu(x):
    return np.maximum(0, x)


def softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(P, y_true):
    B = P.shape[0]
    P = np.clip(P, 1e-15, 1.0 - 1e-15)
    correct_logprobs = -np.log(P[np.arange(B), y_true])
    return np.sum(correct_logprobs) / B


def predict(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)
    return np.argmax(P, axis=1)


# 4. 训练循环 (Mini-batch) & 手写反向传播
print("\n=== 3. 开始 Mini-batch 训练 ===")
epochs = 100
batch_size = 64
learning_rate = 0.1

train_losses = []
val_accuracies = []
printed_gradients_shape = False

for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for i in range(num_batches):
        # 提取 mini-batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train.shape[0])
        Xb = X_train_shuffled[start_idx:end_idx]
        yb = y_train_shuffled[start_idx:end_idx]
        B = Xb.shape[0]  # 当前 batch 的实际样本数

        # --- 前向传播 (Forward) ---
        Z1 = Xb @ W1 + b1  # (B, H)
        A1 = relu(Z1)  # (B, H)
        Z2 = A1 @ W2 + b2  # (B, C)
        P = softmax(Z2)  # (B, C)

        loss = compute_loss(P, yb)
        epoch_loss += loss * B

        # --- 反向传播 (Backward) ---
        # 1. 计算输出层梯度 (Softmax + Cross Entropy 联合求导)
        dZ2 = P.copy()
        dZ2[np.arange(B), yb] -= 1  # (B, C)  原理: 预测概率 - 真实标签(One-hot)
        # 2. 计算隐藏层到输出层参数梯度
        dW2 = (1.0 / B) * (A1.T @ dZ2)  # (H, C)
        db2 = (1.0 / B) * np.sum(dZ2, axis=0)  # (C,)
        # 3. 计算隐藏层误差 dA1 和 dZ1
        dA1 = dZ2 @ W2.T  # (B, H)
        dZ1 = dA1.copy()  # ReLU 的导数：当 Z1 > 0 时为 1，否则为 0
        dZ1[Z1 <= 0] = 0  # (B, H)
        # 4. 计算输入层到隐藏层参数梯度
        dW1 = (1.0 / B) * (Xb.T @ dZ1)  # (D, H)
        db1 = (1.0 / B) * np.sum(dZ1, axis=0)  # (H,)

        # 实验要求：打印梯度的 shape (仅第一轮第一个batch打印)
        if not printed_gradients_shape:
            print(f"反向传播梯度形状打印:")
            print(f"dW1 shape: {dW1.shape}, db1 shape: {db1.shape}")
            print(f"dW2 shape: {dW2.shape}, db2 shape: {db2.shape}\n")
            printed_gradients_shape = True

        # --- 更新参数 (Gradient Descent) ---
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # 计算平均 loss
    avg_epoch_loss = epoch_loss / X_train.shape[0]
    train_losses.append(avg_epoch_loss)

    # 计算验证集 Accuracy
    val_preds = predict(X_val, W1, b1, W2, b2)
    val_acc = np.mean(val_preds == y_val)
    val_accuracies.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:03d}/{epochs} - Loss: {avg_epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

# 5. 最终指标评估
train_preds = predict(X_train, W1, b1, W2, b2)
test_preds = predict(X_test, W1, b1, W2, b2)

print(f"Train Accuracy: {np.mean(train_preds == y_train):.4%}")
print(f"Val Accuracy:   {np.mean(val_preds == y_val):.4%}")
print(f"Test Accuracy:  {np.mean(test_preds == y_test):.4%}")

# 6. 绘制 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, color='coral', linewidth=2, label='Train Loss')
plt.title('2-Layer MLP Training Loss (Mini-batch)')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('lab3_loss_curve.png')
plt.show()