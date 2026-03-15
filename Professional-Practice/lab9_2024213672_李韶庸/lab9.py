import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 1. 核心算子：缩放点积注意力
printed_attention_info = False  # 全局标志，确保只打印一次


def scaled_dot_product_attention(Q, K, V, mask=None):
    global printed_attention_info

    d_k = Q.size(-1)

    # 1. 计算 Q 和 K^T 的点积，并缩放
    # K.transpose(-2, -1) 将形状变为 (B, d_k, Tk)
    # scores: (B, Tq, d_k) @ (B, d_k, Tk) -> (B, Tq, Tk)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码处理 (Masking)
    if mask is not None:
        # 将 mask 中为 0 (False) 的位置替换为极大负数，Softmax 后就会接近 0
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 归一化注意力权重
    attn = F.softmax(scores, dim=-1)  # (B, Tq, Tk)

    # 4. 乘以 V 得到最终输出
    # out: (B, Tq, Tk) @ (B, Tk, d_v) -> (B, Tq, d_v)
    out = torch.matmul(attn, V)

    # 根据实验要求，首次调用时打印维度和分布信息
    if not printed_attention_info:
        print("\n[注意力底层计算自检]")
        print(f"  Out shape: {list(out.shape)}")
        print(f"  Attn shape: {list(attn.shape)}")
        # 验证 softmax 沿最后维度的和是否为 1
        sum_attn = attn.sum(dim=-1)[0, 0].item()
        print(f"  attn.sum(dim=-1) (选取样本0): {sum_attn:.4f} (应极度接近1.0)")
        printed_attention_info = True

    return out, attn

# 2. 合成数据生成 (指针检索任务)
class PointerRetrievalDataset(Dataset):
    def __init__(self, num_samples=5000, vocab_size=20, seq_len=16):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        # 随机生成长度为 seq_len 的 token 序列 x
        self.X = torch.randint(0, vocab_size, (num_samples, seq_len))
        # 随机生成指针位置 p (0 到 seq_len - 1)
        self.P = torch.randint(0, seq_len, (num_samples,))
        # 目标 y 为 x 在指针位置 p 上的 token
        self.Y = torch.zeros(num_samples, dtype=torch.long)
        for i in range(num_samples):
            self.Y[i] = self.X[i, self.P[i]]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.P[idx], self.Y[idx]

# 3. 构建“指针检索”注意力模型
class PointerAttentionModel(nn.Module):
    def __init__(self, vocab_size=20, seq_len=16, d_model=32):
        super(PointerAttentionModel, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. 序列 Token 的 Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置 Embedding：让 K 能携带位置信息，与 Q（位置查询）对齐
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        # 2. 注意力投影矩阵
        # K 基于位置 Embedding，V 基于 Token Embedding
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        # 针对于指针 one-hot 产生 Q
        self.Wq = nn.Linear(seq_len, d_model)

        # 3. 最终分类器
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, p):
        """
        x: (B, L) 序列 token IDs
        p: (B,) 指针位置
        """
        B = x.size(0)

        # 1. 构造 K（基于位置 Embedding）, V（基于 Token Embedding）
        # 位置 id: (B, L)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)   # (B, L, d_model)
        x_emb = self.embedding(x)               # (B, L, d_model)
        K = self.Wk(pos_emb)  # K 编码位置信息，与 Q（位置查询）语义对齐
        V = self.Wv(x_emb)    # V 编码 token 身份，用于读取目标 token

        # 2. 构造 Q (基于指针 One-Hot 特征)
        # q_pos: (B,) -> (B, L)
        q_pos = F.one_hot(p, num_classes=self.seq_len).float()
        # 增加一个时间维度 Tq = 1 -> (B, 1, L)
        q_pos = q_pos.unsqueeze(1)
        # 通过线性层映射得到 Q: (B, 1, L) @ (L, d_model) -> (B, 1, d_model)
        Q = self.Wq(q_pos)

        # 3. 缩放点积注意力计算 (这里无需 mask)
        # out: (B, 1, d_model), attn: (B, 1, L)
        out, attn = scaled_dot_product_attention(Q, K, V)

        # 4. 分类器
        # 挤压时间维度 (B, 1, d_model) -> (B, d_model)
        out = out.squeeze(1)
        logits = self.classifier(out)  # (B, vocab_size)

        return logits, attn

# 4. 训练 Pipeline
def train_model():
    print("=== 1. 初始化数据与模型 ===")
    vocab_size, seq_len = 20, 16
    train_dataset = PointerRetrievalDataset(num_samples=4000, vocab_size=vocab_size, seq_len=seq_len)
    test_dataset = PointerRetrievalDataset(num_samples=1000, vocab_size=vocab_size, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PointerAttentionModel(vocab_size=vocab_size, seq_len=seq_len, d_model=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n=== 2. 开始训练 ===")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, p, y in train_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x, p)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 测试阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, p, y in test_loader:
                x, p, y = x.to(device), p.to(device), y.to(device)
                logits, _ = model(x, p)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {running_loss / len(train_loader):.4f} | Test Acc: {correct / total:.2%}")

    return model, test_dataset, device

# 5. 可视化注意力权重
def visualize_attention(model, dataset, device):
    print("\n=== 3. 验证注意力权重 (Attention Analysis) ===")
    model.eval()

    # 随机抽取一个样本
    sample_idx = np.random.randint(0, len(dataset))
    x_val, p_val, y_val = dataset[sample_idx]

    # 增加 Batch 维度进行推理
    x_input = x_val.unsqueeze(0).to(device)
    p_input = p_val.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn = model(x_input, p_input)

    # 获取注意力权重数组 (剔除 Batch 和 Tq 维度)
    attn_weights = attn[0, 0, :].cpu().numpy()
    pred_y = logits.argmax(dim=1).item()
    p_true = p_val.item()

    print(f"随机样本索引: {sample_idx}")
    print(f"输入序列 (x): {x_val.tolist()}")
    print(f"真实指针位置 (p): {p_true}")
    print(f"目标 Token (y): {y_val.item()}")
    print(f"模型预测输出: {pred_y}")
    print(f"注意力的最大索引 argmax(attn): {np.argmax(attn_weights)}")

    print("\n[注意力权重分布]:")
    for i, w in enumerate(attn_weights):
        flag = " <--- 指针目标!" if i == p_true else ""
        print(f" 位置 {i:02d}: {w:.4f}{flag}")

    # 检查注意力是否精准聚焦在位置 p
    if np.argmax(attn_weights) == p_true:
        print("\n>>> 结论：注意力机制成功聚焦到了指针位置 (argmax(attn) == p)！")
    else:
        print(f"\n>>> 注意：注意力最大值在位置 {np.argmax(attn_weights)}，真实指针在位置 {p_true}。")

    # 绘制注意力权重柱状图
    seq_len = len(attn_weights)
    colors = ['tomato' if i == p_true else 'steelblue' for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(seq_len), attn_weights, color=colors)
    ax.set_xticks(range(seq_len))
    ax.set_xlabel("序列位置")
    ax.set_ylabel("注意力权重")
    ax.set_title(f"注意力权重分布  (真实指针 p={p_true}, 预测 y={pred_y}, 目标 y={y_val.item()})")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # 标注红色柱（指针位置）
    ax.bar(p_true, attn_weights[p_true], color='tomato', label=f'指针位置 p={p_true}')
    ax.legend()

    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=150)
    plt.show()
    print("注意力权重图已保存至 attention_weights.png")


if __name__ == '__main__':
    # 训练模型
    trained_model, test_dataset, dev = train_model()
    # 验证注意力并可视化
    visualize_attention(trained_model, test_dataset, dev)