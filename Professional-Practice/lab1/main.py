import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 载入并处理数据
digits = load_digits()
X, y = digits.data, digits.target

# 划分比例：训练 70%, 验证 15%, 测试 15%
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"数据划分完成: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")


# 2. 转换为 Tensor 并创建 DataLoader
def to_tensor_loader(X, y, batch_size, shuffle=False):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = to_tensor_loader(X_train, y_train, batch_size=64, shuffle=True)
val_loader = to_tensor_loader(X_val, y_val, batch_size=256)
test_loader = to_tensor_loader(X_test, y_test, batch_size=256)


# 3. 定义 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
epochs = 20
train_losses = []

for epoch in range(epochs):
    model.train()  # 切换训练模式
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()  # 1. 清空梯度
        outputs = model(batch_x)  # 2. 前向传播
        loss = criterion(outputs, batch_y)  # 3. 计算 Loss
        loss.backward()  # 4. 反向传播
        optimizer.step()  # 5. 更新权重
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 每个 Epoch 结束后做 Validation
    model.eval()  # 切换评估模式
    correct = 0
    with torch.no_grad():  # 验证时不计算梯度
        for bx, by in val_loader:
            outputs = model(bx)
            preds = outputs.argmax(dim=1)
            correct += (preds == by).sum().item()

    val_acc = correct / len(X_val)
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# 5. 测试与保存
model.eval()
test_correct = 0
with torch.no_grad():
    for bx, by in test_loader:
        test_correct += (model(bx).argmax(dim=1) == by).sum().item()
print(f"\nFinal Test Accuracy: {test_correct / len(X_test):.4f}")

# 保存权重
torch.save(model.state_dict(), 'mlp_model.pth')

# 加载并再次验证
new_model = SimpleMLP()
new_model.load_state_dict(torch.load('mlp_model.pth'))
new_model.eval()
# (此处再次计算 acc 的代码同上)

# 6. 绘图
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Train Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('training_loss_curve—lab1.png')