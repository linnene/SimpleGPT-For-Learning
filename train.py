import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import os
import urllib.request
from tqdm import tqdm

# --- 超参数设置 ---
batch_size = 32 # 多少个独立的序列并行处理?
block_size = 64 # 上下文的最大长度是多少?
max_iters = 10000 # 训练迭代次数
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256 # 嵌入维度
n_head = 6 # 注意力头的数量
n_layer = 4 # Transformer 层的数量
dropout = 0.2
# ------------------

print(f"Using device: {device}")

torch.manual_seed(1337)

# --- 数据准备 ---
# 下载 Tiny Shakespeare 数据集
file_path = 'input.txt'
if not os.path.exists(file_path):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Creating a dummy dataset instead.")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Hello world! This is a simple GPT model. " * 1000)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 这里是文本中出现的所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Chars: {''.join(chars)}")

# 创建从字符到整数的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码器: 字符串 -> 整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码器: 整数列表 -> 字符串

# 训练和测试集划分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 前 90% 是训练集，其余是验证集
train_data = data[:n]
val_data = data[n:]

# 数据加载器
def get_batch(split):
    # 生成一小批数据输入 x 和目标 y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 估算损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 初始化模型 ---
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
model = model.to(device)

# 打印模型参数数量
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 训练循环 ---
print("Starting training...")
pbar = tqdm(range(max_iters), desc="Training")
for iter in pbar:

    # 每隔一段时间评估一次损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        pbar.set_postfix({'train_loss': f"{losses['train']:.4f}", 'val_loss': f"{losses['val']:.4f}"})

    # 采样一批数据
    xb, yb = get_batch('train')

    # 评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete.")

# --- 保存模型 ---
print("Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'vocab_size': vocab_size
}, 'gpt_model.pth')
print("Model saved to gpt_model.pth")

# --- 生成示例 ---
print("Generating sample text:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
