import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据及预处理
data = load_breast_cancer()
X, y = data.data, data.target
print(f"原始数据 shape: X={X.shape}, y={y.shape}")

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 实现 sigmoid 函数
def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))
#打印 sigmiod（0）
print(f"验证 sigmoid(0) = {sigmoid(0)}")

# 3. 实现预测概率函数 p = σ(Xw + b)
def predict_prob(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

# 4. 实现二元交叉熵损失函数 (Log Loss)
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    # 增加微小偏移量 epsilon，防止 log(0) 导致无穷大 (NaN)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = - (1.0 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 5. 实现梯度计算函数
def compute_gradients(X, y_true, y_pred):
    m = X.shape[0]
    dz = y_pred - y_true  # 预测值与真实值的误差

    dw = (1.0 / m) * np.dot(X.T, dz)  # 权重 w 的梯度，形状为 (n,)
    db = (1.0 / m) * np.sum(dz)  # 偏置 b 的梯度，标量
    return dw, db

# 6. 实现训练循环 (梯度下降)
# 初始化参数
n_samples, n_features = X_train.shape
w = np.zeros(n_features)  # 权重初始化为 0
b = 0.0  # 偏置初始化为 0

# 超参数设置
learning_rate = 0.1
epochs = 500
losses = []

for epoch in range(epochs):
    # 1. 前向传播：计算预测概率
    p = predict_prob(X_train, w, b)

    # 2. 计算并记录 Loss
    loss = compute_loss(y_train, p)
    losses.append(loss)

    # 3. 反向传播：计算梯度
    dw, db = compute_gradients(X_train, y_train, p)

    # 4. 梯度下降：更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 每 50 轮打印一次信息
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1:03d}/{epochs}], Loss: {loss:.4f}")

# 7. 在 Test 集计算 Accuracy
test_probs = predict_prob(X_test, w, b)

# 概率 >= 0.5 判为正例(1)，否则判为负例(0)
y_pred_class = (test_probs >= 0.5).astype(int)

# 计算准确率 (预测正确的数量 / 总测试样本数)
accuracy = np.mean(y_pred_class == y_test)
print(f"最终测试集准确率 (Test Accuracy): {accuracy:.4%}")

# 8. 画出 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, color='blue', linewidth=2)
plt.title('Logistic Regression Training Loss (From Scratch)')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('lab2_loss_curve.png')
plt.show()