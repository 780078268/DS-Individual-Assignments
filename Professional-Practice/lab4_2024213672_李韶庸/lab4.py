import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据及预处理
X, y = load_digits(return_X_y=True)

# 划分训练集和测试集 (80% 训练, 20% 测试)，并使用 stratify 保证类别分布一致
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 从剩余的 80% 中再划分出 25% 作为验证集 (即总体数据的 20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# 初始化 StandardScaler，并在训练集上进行 fit，然后应用到所有数据集
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"数据划分完成: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}");
# 2. 定义带 Dropout 和 Momentum 的 MLP 类
# ==========================================
class TwoLayerMLP:
    def __init__(self, D=64, H=128, C=10, dropout_p=0.0, gamma=0.0, lr=0.05):
        self.p = dropout_p  # Dropout 层丢弃神经元的概率
        self.gamma = gamma  # Momentum 优化器的动量系数
        self.lr = lr  # 学习率

        # 1. 权重初始化: 使用标准正态分布 N(0, 0.01)，偏置初始化为 0
        np.random.seed(42)  # 固定随机种子，确保对照实验的起点一致
        self.W1 = np.random.randn(D, H) * 0.01
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(H, C) * 0.01
        self.b2 = np.zeros(C)

        # 2. 速度变量初始化: 形状与对应参数一致，初始值为 0 (用于 Momentum SGD)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

        # 缓存字典，用于在 forward 阶段保存变量，供 backward 阶段使用
        self.cache = {}

    def forward(self, X, mode='train'):
        """
        前向传播函数。
        根据 mode 的不同，决定是否应用 Dropout。
        """
        # 第一层: 线性变换 + ReLU 激活
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU

        # --- Dropout Layer (仅在 Hidden 层激活后应用) ---
        mask = None
        # Inverted Dropout: 仅在 train 模式下，且丢弃概率 p > 0 时激活
        if self.p > 0 and mode == 'train':
            keep_prob = 1.0 - self.p
            # 生成随机掩码矩阵，并除以 keep_prob 进行缩放 (Inverted Dropout 的核心)
            mask = (np.random.rand(*A1.shape) < keep_prob) / keep_prob
            A1_drop = A1 * mask
        else:
            # eval 模式下或 p=0 时，不进行任何丢弃或缩放操作
            A1_drop = A1

        # 第二层: 线性变换
        Z2 = np.dot(A1_drop, self.W2) + self.b2

        # Softmax 激活函数 (处理输出)
        # 为防止指数爆炸，先减去每行的最大值，提高数值稳定性
        Z2_shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
        exp_Z2 = np.exp(Z2_shifted)
        P = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

        # 如果在训练模式，缓存必要的中间变量以备反向传播使用
        if mode == 'train':
            self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'mask': mask, 'A1_drop': A1_drop, 'P': P}

        return P

    def compute_loss(self, P, y_true):
        """
        计算多分类交叉熵损失 (Log Loss)
        """
        B = P.shape[0]
        # 添加极小值 epsilon 防止 log(0) 产生 NaN
        epsilon = 1e-15
        P_clipped = np.clip(P, epsilon, 1.0 - epsilon)
        # 提取真实类别对应的预测概率并计算平均对数损失
        return -np.sum(np.log(P_clipped[np.arange(B), y_true])) / B

    def backward(self, y_true):
        """
        反向传播函数，计算梯度并使用 Momentum SGD 规则更新参数。
        """
        # 从缓存中取出前向传播时的变量
        X = self.cache['X']
        Z1 = self.cache['Z1']
        mask = self.cache['mask']
        A1_drop = self.cache['A1_drop']
        P = self.cache['P']
        B = X.shape[0]  # 当前批次的样本数

        # 1. 计算输出层梯度 (Softmax 与交叉熵结合求导)
        dZ2 = P.copy()
        dZ2[np.arange(B), y_true] -= 1  # 预测概率减去真实标签 (one-hot形式)

        # 计算第二层参数的梯度
        dW2 = (1.0 / B) * np.dot(A1_drop.T, dZ2)
        db2 = (1.0 / B) * np.sum(dZ2, axis=0)

        # 2. 计算隐藏层误差，回传至第一层
        dA1_drop = np.dot(dZ2, self.W2.T)

        # --- 穿过 Dropout 层 ---
        # 如果前向传播时使用了 Dropout 掩码，误差只能回传给未被丢弃的神经元
        if mask is not None:
            dA1 = dA1_drop * mask
        else:
            dA1 = dA1_drop

        # --- 穿过 ReLU 层 ---
        dZ1 = dA1.copy()
        dZ1[Z1 <= 0] = 0  # ReLU 导数：输入 <= 0 时梯度为 0，否则为 1

        # 计算第一层参数的梯度
        dW1 = (1.0 / B) * np.dot(X.T, dZ1)
        db1 = (1.0 / B) * np.sum(dZ1, axis=0)

        # 3. 参数更新 (Momentum SGD)
        # 更新规则:
        # V_{new} = gamma * V_{old} + lr * dW
        # W_{new} = W_{old} - V_{new}

        self.v_W2 = self.gamma * self.v_W2 + self.lr * dW2
        self.W2 -= self.v_W2

        self.v_b2 = self.gamma * self.v_b2 + self.lr * db2
        self.b2 -= self.v_b2

        self.v_W1 = self.gamma * self.v_W1 + self.lr * dW1
        self.W1 -= self.v_W1

        self.v_b1 = self.gamma * self.v_b1 + self.lr * db1
        self.b1 -= self.v_b1

    def predict(self, X):
        """
        预测函数。
        强制使用 'eval' 模式进行前向传播，返回概率最高的类别索引。
        """
        P = self.forward(X, mode='eval')
        return np.argmax(P, axis=1)


# 3. 训练循环函数
def train_model(model, name, epochs=100, batch_size=64):
    print(f"\n=== 开始训练: {name} ===")
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))

        for i in range(num_batches):
            # 提取当前的 Mini-batch 数据
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_train.shape[0])
            Xb = X_shuffled[start_idx:end_idx]
            yb = y_shuffled[start_idx:end_idx]

            # --- 1. 前向传播 (Train 模式) ---
            P = model.forward(Xb, mode='train')

            # --- 2. 计算并累计 Loss ---
            loss = model.compute_loss(P, yb)
            # 乘以当前批次的实际大小，以便后续计算精准的平均 loss
            epoch_loss += loss * Xb.shape[0]

            # --- 3. 反向传播与参数更新 ---
            model.backward(yb)

        # 记录当前 Epoch 的平均训练损失
        avg_loss = epoch_loss / X_train.shape[0]
        train_losses.append(avg_loss)

        # 在验证集上评估模型性能 (强制使用 eval 模式)
        val_preds = model.predict(X_val)
        val_acc = np.mean(val_preds == y_val)
        val_accuracies.append(val_acc)

        # 每隔一定轮数打印训练进度
        if (epoch + 1) % 20 == 0:
            print(f"[{name}] Epoch {epoch + 1:03d}/{epochs} - Train Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    # 训练结束后，在独立的测试集上评估最终性能
    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == y_test)
    print(f"{name} - 最终测试集准确率 (Test Accuracy): {test_acc:.4%}")

    return train_losses, val_accuracies

# 4. 实例化模型并进行对比训练
total_epochs = 80

# 模型 A (Baseline): 使用普通的 SGD (无动量，gamma=0)，且不启用 Dropout (p=0)
model_A = TwoLayerMLP(dropout_p=0.0, gamma=0.0, lr=0.05)
losses_A, val_acc_A = train_model(model_A, "Model A (Baseline)", epochs=total_epochs)

# 模型 B (Improved): 启用 Momentum SGD (动量系数 gamma=0.9)，并开启 Dropout (概率 p=0.5)
model_B = TwoLayerMLP(dropout_p=0.5, gamma=0.9, lr=0.05)
losses_B, val_acc_B = train_model(model_B, "Model B (Improved)", epochs=total_epochs)

# 5. 绘制并保存对比结果曲线
plt.figure(figsize=(10, 6))

# 绘制模型 A 和 B 的训练损失曲线进行直观对比
plt.plot(range(1, total_epochs + 1), losses_A, color='blue', linestyle='-', linewidth=2,
         label='Model A (Standard SGD, No Dropout)')
plt.plot(range(1, total_epochs + 1), losses_B, color='red', linestyle='-', linewidth=2,
         label='Model B (Momentum=0.9, Dropout=0.5)')

plt.title('Training Loss Comparison: Baseline vs. Improved Architecture')
plt.xlabel('Epochs')
plt.ylabel('Average Log Loss (Cross-Entropy)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('lab4_comparison_curve.png')
plt.show()