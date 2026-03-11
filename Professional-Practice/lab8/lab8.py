import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
import re
from collections import Counter

categories = ['rec.autos', 'sci.space', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

all_texts = newsgroups.data
all_labels = newsgroups.target
selected_texts, selected_labels = [], []
counts = {c: 0 for c in range(len(categories))}
for text, label in zip(all_texts, all_labels):
    if counts[label] < 500:
        selected_texts.append(text)
        selected_labels.append(label)
        counts[label] += 1

X_train_val, X_test, y_train_val, y_test = train_test_split(selected_texts, selected_labels, test_size=200,
                                                            random_state=42, stratify=selected_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=200, random_state=42,
                                                  stratify=y_train_val)

print(f"数据划分: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# 2. 构建词表与分词 (Tokenizer)


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return text.split()


# 统计训练集词频
word_counts = Counter()
for text in X_train:
    word_counts.update(tokenize(text))

K = 5000  # 词表总大小
# 预留 0 给 <pad> (用于后续 Embedding), 预留 1 给 <unk>
most_common_words = word_counts.most_common(K - 2)
vocab = {'<pad>': 0, '<unk>': 1}
for word, _ in most_common_words:
    vocab[word] = len(vocab)

itos = {i: w for w, i in vocab.items()}
print(f"Vocab Size 词表大小: {len(vocab)} (Top-{K})")


def encode_texts(texts):
    return [[vocab.get(w, 1) for w in tokenize(t)] for t in texts]


train_ids = encode_texts(X_train)
val_ids = encode_texts(X_val)
test_ids = encode_texts(X_test)

sample_text = "I love space and graphics!"
sample_tokens = tokenize(sample_text)
sample_ids = [vocab.get(w, 1) for w in sample_tokens]
print(f"示例文本: '{sample_text}'")
print(f"Tokens: {sample_tokens}")
print(f"IDs: {sample_ids}")

# 3. 实现 BoW 特征 (简化 TF-IDF + L2 归一化)
df = np.zeros(K)
for ids in train_ids:
    unique_ids = set(ids)
    for w_id in unique_ids:
        df[w_id] += 1
N_train = len(train_ids)
idf = np.log(N_train / (1.0 + df))

def make_bow_matrix(id_lists):
    bow = np.zeros((len(id_lists), K), dtype=np.float32)
    for i, ids in enumerate(id_lists):
        if len(ids) == 0: continue
        tf = Counter(ids)
        length = len(ids)
        for w_id, count in tf.items():
            bow[i, w_id] = (count / length) * idf[w_id]
        norm = np.linalg.norm(bow[i])
        if norm > 0:
            bow[i] = bow[i] / norm
    return bow


X_train_bow = torch.tensor(make_bow_matrix(train_ids))
X_val_bow = torch.tensor(make_bow_matrix(val_ids))
X_test_bow = torch.tensor(make_bow_matrix(test_ids))

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

print(f"Train BoW 维度: {X_train_bow.shape}")

# 4. 从零实现 BoW + Softmax 分类器
class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        # 手写权重和偏置：logits = x @ W + b
        self.W = nn.Parameter(torch.randn(vocab_size, num_classes) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return x @ self.W + self.b
device = torch.device("mps")

bow_model = BoWClassifier(vocab_size=K, num_classes=len(categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bow_model.parameters(), lr=1e-3)

epochs = 100  # 因为是 full-batch, 稍微增加 epoch 数以保证收敛
X_train_bow_d, y_train_d = X_train_bow.to(device), y_train_t.to(device)
X_val_bow_d, y_val_d = X_val_bow.to(device), y_val_t.to(device)

for epoch in range(epochs):
    bow_model.train()
    optimizer.zero_grad()
    logits = bow_model(X_train_bow_d)
    loss = criterion(logits, y_train_d)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        bow_model.eval()
        with torch.no_grad():
            val_logits = bow_model(X_val_bow_d)
            val_acc = (val_logits.argmax(dim=1) == y_val_d).float().mean().item()
        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4%}")

bow_model.eval()
with torch.no_grad():
    test_logits = bow_model(X_test_bow.to(device))
    test_acc_bow = (test_logits.argmax(dim=1) == y_test_t.to(device)).float().mean().item()
print(f"BoW 模型最终 Test Acc: {test_acc_bow:.4%}")

# 5. BoW 可解释性分析 (Top-10 Words)
print("\n=== 5. BoW 可解释性分析 (W 权重解析) ===")
W_learned = bow_model.W.detach().cpu().numpy()

for class_idx, class_name in enumerate(categories):
    # 对 W 的第 class_idx 列进行降序排序，获取 top 10 的词汇索引
    # np.argsort 返回升序索引，[::-1] 翻转为降序
    top_indices = np.argsort(W_learned[:, class_idx])[::-1][:10]
    top_words = [itos[idx] for idx in top_indices]
    print(f"类别 [{class_name}] 的 Top-10 支撑词汇: {top_words}")

# 6. 实现 Embedding 平均池化表示
class EmbedBagClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        # padding_idx=0 使得 <pad> 词向量始终为 0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        # x: (B, T), mask: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        # 利用 mask 消除 pad 符号的影响
        # mask.unsqueeze(-1) 将形状变为 (B, T, 1)，利用广播机制掩盖 pad 的词向量
        sum_emb = (emb * mask.unsqueeze(-1)).sum(dim=1)  # (B, E)
        # 计算每个句子的真实长度，clamp 防止除以 0
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        # 求平均
        e_bar = sum_emb / lengths  # (B, E)
        return self.fc(e_bar)


# 变长序列 Padding & Masking (自定义 collate_fn)
def collate_fn(batch):
    ids_list, labels = zip(*batch)
    # 找到 batch 内的最长序列
    max_len = max(len(seq) for seq in ids_list) if ids_list else 0
    # 防止空文本，强制 min max_len = 1
    max_len = max(max_len, 1)

    padded_ids = np.zeros((len(ids_list), max_len), dtype=np.int64)
    mask = np.zeros((len(ids_list), max_len), dtype=np.float32)

    for i, seq in enumerate(ids_list):
        curr_len = len(seq)
        if curr_len > 0:
            padded_ids[i, :curr_len] = seq
            mask[i, :curr_len] = 1.0  # 真实词语的位置置为 1.0

    return torch.tensor(padded_ids), torch.tensor(mask), torch.tensor(labels, dtype=torch.long)


class SeqDataset(Dataset):
    def __init__(self, data_ids, labels):
        self.data_ids = data_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_ids[idx], self.labels[idx]


train_loader = DataLoader(SeqDataset(train_ids, y_train), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(SeqDataset(val_ids, y_val), batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(SeqDataset(test_ids, y_test), batch_size=64, shuffle=False, collate_fn=collate_fn)

emb_model = EmbedBagClassifier(vocab_size=K, embed_dim=64, num_classes=len(categories)).to(device)
emb_optimizer = optim.Adam(emb_model.parameters(), lr=1e-3)

epochs_emb = 15
for epoch in range(epochs_emb):
    emb_model.train()
    running_loss = 0.0
    for bx, bmask, by in train_loader:
        bx, bmask, by = bx.to(device), bmask.to(device), by.to(device)
        emb_optimizer.zero_grad()
        logits = emb_model(bx, bmask)
        loss = criterion(logits, by)
        loss.backward()
        emb_optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 3 == 0:
        emb_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, bmask, by in val_loader:
                bx, bmask, by = bx.to(device), bmask.to(device), by.to(device)
                logits = emb_model(bx, bmask)
                correct += (logits.argmax(dim=1) == by).sum().item()
                total += by.size(0)
        print(
            f"Epoch [{epoch + 1:02d}/{epochs_emb}] | Train Loss: {running_loss / len(train_loader):.4f} | Val Acc: {correct / total:.4%}")

emb_model.eval()
test_correct = 0
with torch.no_grad():
    for bx, bmask, by in test_loader:
        bx, bmask, by = bx.to(device), bmask.to(device), by.to(device)
        logits = emb_model(bx, bmask)
        test_correct += (logits.argmax(dim=1) == by).sum().item()
print(f"Embedding 平均池化模型最终 Test Acc: {test_correct / len(test_loader.dataset):.4%}")