import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# 1. 数据读取与字符编码
file_path = 'tiny_corpus_rnn.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    TEXT = f.read()

# 构建字符表和映射字典
vocab = sorted(set(TEXT))
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
ids = [stoi[ch] for ch in TEXT]


# 构造 Dataset
class CharDataset(Dataset):
    def __init__(self, data_ids, seq_len=32):
        self.data_ids = data_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_ids) - self.seq_len

    def __getitem__(self, idx):
        # x: ids[i : i+T]
        # y: ids[i+1 : i+T+1]
        x_chunk = self.data_ids[idx: idx + self.seq_len]
        y_chunk = self.data_ids[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(x_chunk, dtype=torch.long), torch.tensor(y_chunk, dtype=torch.long)


T = 32
batch_size = 64
dataset = CharDataset(ids, seq_len=T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 2. 定义手写 RNN 模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.E = embed_size
        self.H = hidden_size
        self.emb = nn.Embedding(self.vocab_size, self.E)
        self.Wxh = nn.Parameter(torch.randn(self.E, self.H) * 0.01)
        self.Whh = nn.Parameter(torch.randn(self.H, self.H) * 0.01)
        self.bh = nn.Parameter(torch.zeros(self.H))

        # 3. 输出层
        self.fc = nn.Linear(self.H, self.vocab_size)

    def my_rnn_cell(self, x_t, h_prev):
        B, E_dim = x_t.shape
        _, H_dim = h_prev.shape
        assert E_dim == self.Wxh.shape[0], "输入特征维度 E 错误！"
        assert H_dim == self.Whh.shape[0], "隐藏状态维度 H 错误！"
        h_t = torch.tanh(x_t @ self.Wxh + h_prev @ self.Whh + self.bh)
        assert h_t.shape == (B, self.H), "Cell 输出维度错误！"
        return h_t

    def forward(self, x):
        B, seq_T = x.shape
        x_emb = self.emb(x)
        assert x_emb.shape == (B, seq_T, self.E), "Embedding 输出维度错误！"
        h = torch.zeros(B, self.H, device=x.device)
        H_seq = []
        for t in range(seq_T):
            h = self.my_rnn_cell(x_emb[:, t, :], h)
            H_seq.append(h)
        H_seq = torch.stack(H_seq, dim=1)  # (B, T, H)
        assert H_seq.shape == (B, seq_T, self.H), "Unroll 后隐藏序列维度错误！"
        logits = self.fc(H_seq)  # (B, T, V)
        return logits

# 3. 文本生成采样函数
def sample(model, seed_text, gen_len=200, temperature=1.0, device='cpu'):
    model.eval()
    with torch.no_grad():
        generated_ids = [stoi[c] for c in seed_text]
        for _ in range(gen_len):
            x_input = torch.tensor([generated_ids], dtype=torch.long).to(device)
            logits = model(x_input)  # shape: (1, L, V)
            last_step_logits = logits[0, -1, :]  # shape: (V,)
            probs = torch.softmax(last_step_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)
    generated_text = "".join([itos[i] for i in generated_ids])
    return generated_text

# 4. 主程序：训练 Pipeline
if __name__ == '__main__':
    print("\n=== 2. 模型维度自检 ===")
    device = torch.device("mps")
    E, H = 32, 128
    model = CharRNN(vocab_size=vocab_size, embed_size=E, hidden_size=H).to(device)
    dummy_x = torch.randint(0, vocab_size, (batch_size, T)).to(device)
    dummy_logits = model(dummy_x)
    print(f"随机 Batch 输入 shape: {dummy_x.shape}")
    print(f"Logits 输出 shape: {dummy_logits.shape} (预期为 [64, 32, {vocab_size}])")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 15
    history_loss = []
    print(f"\n[训练前瞎猜]:\n{sample(model, seed_text='The ', gen_len=50, device=device)}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)  # (B, T, V)

            # 损失计算：将 (B, T, V) 展平为 (B*T, V)，将 y 从 (B, T) 展平为 (B*T,)
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))

            loss.backward()

            # RNN 的灵魂操作：梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        history_loss.append(avg_loss)
        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {avg_loss:.4f}")

    final_text = sample(model, seed_text="人工智能", gen_len=150, temperature=0.8, device=device)
    print(f"生成结果:\n{final_text}\n")

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history_loss, marker='o', color='purple')
    plt.title('CharRNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.savefig('lab7_loss_curve.png')
    plt.show()