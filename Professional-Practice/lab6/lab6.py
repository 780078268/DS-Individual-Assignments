import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 准备数据及预处理 (FashionMNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 封装为 DataLoader
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# 2. 定义 LeNet-5 模型结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 400 -> 120
            nn.ReLU(),
            nn.Linear(120, 84),  # 120 -> 84
            nn.ReLU(),
            nn.Linear(84, 10)  # 84 -> 10 (10个类别)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# 3. 维度自检函数
def check_shape():
    print("\n=== 2. 维度自检 (Shape Check) ===")
    model = LeNet5()
    # 模拟一个 Batch Size = 1 的单通道 28x28 图像
    x = torch.randn(1, 1, 28, 28)
    print(f"初始输入 Input shape: \t {list(x.shape)}")

    # 逐层跟踪打印并验证
    x = model.layer1[0](x)  # Conv2d
    print(f"Layer 1 - Conv2d: \t {list(x.shape)}")
    assert list(x.shape) == [1, 6, 28, 28], "Layer 1 Conv 维度错误"

    x = model.layer1[2](model.layer1[1](x))  # ReLU + MaxPool2d
    print(f"Layer 1 - MaxPool: \t {list(x.shape)}")
    assert list(x.shape) == [1, 6, 14, 14], "Layer 1 Pool 维度错误"

    x = model.layer2[0](x)  # Conv2d
    print(f"Layer 2 - Conv2d: \t {list(x.shape)}")
    assert list(x.shape) == [1, 16, 10, 10], "Layer 2 Conv 维度错误"

    x = model.layer2[2](model.layer2[1](x))  # ReLU + MaxPool2d
    print(f"Layer 2 - MaxPool: \t {list(x.shape)}")
    assert list(x.shape) == [1, 16, 5, 5], "Layer 2 Pool 维度错误"

    x = model.flatten(x)
    print(f"Layer 3 - Flatten: \t {list(x.shape)}")
    assert list(x.shape) == [1, 400], "Flatten 维度错误"

    x = model.fc_layers(x)
    print(f"Layer 4 - Output: \t {list(x.shape)}")
    assert list(x.shape) == [1, 10], "Output 维度错误"

# 4. 训练 Pipeline
if __name__ == '__main__':
    print(f"训练集大小: {len(train_set)} | 测试集大小: {len(test_set)}")
    check_shape()
    device = "mps"
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    history_loss = []
    history_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        history_loss.append(avg_train_loss)
        history_acc.append(train_acc)

        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # 5. 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), history_loss, marker='o')
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), history_acc, marker='o', color='orange')
    ax2.set_title("Train Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()