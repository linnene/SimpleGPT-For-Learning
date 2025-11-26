import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import os
import urllib.request
from tqdm import tqdm
import re

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
# 下载数据集
file_path = 'input.txt'

if os.path.exists(file_path):
    print("Dataset found locally.")

if not os.path.exists(file_path):
    print("Downloading dataset...")
    # 尝试下载《格林童话》 (Grimm's Fairy Tales)
    # 这是一个经典的英文文本，词汇量适中，比莎士比亚更接近现代叙事，虽然不是纯对话，但比古英语好懂。
    url = "https://www.gutenberg.org/cache/epub/2591/pg2591.txt" 
    try:
        # 由于 Gutenberg 有时会屏蔽自动化请求，我们尝试伪装 User-Agent
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
        )
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete (Grimm's Fairy Tales).")
    except Exception as e:
        print(f"Failed to download Grimm's Fairy Tales: {e}")
        print("Trying Tiny Shakespeare as fallback...")
        try:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, file_path)
            print("Download complete (Tiny Shakespeare).")
        except Exception as e2:
            print(f"Failed to download fallback: {e2}")
            print("Creating a dummy dataset instead.")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Hello world! This is a simple GPT model. " * 1000)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 分词器 (Word-level Tokenizer)
# 使用正则表达式将文本分割为单词、标点符号和换行符
def tokenize(text):
    return re.findall(r"[\w']+|[.,!?;]|\n", text)

tokens = tokenize(text)
print(f"Total tokens in text: {len(tokens)}")

# 这里是文本中出现的所有唯一 token
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")
# print(f"Chars: {''.join(chars)}") # 不再打印所有字符

# 创建从 token 到整数的映射
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
encode = lambda s: [stoi[c] for c in tokenize(s)] # 编码器: 字符串 -> 整数列表
decode = lambda l: ' '.join([itos[i] for i in l]).replace(' \n ', '\n').replace(' .', '.').replace(' ,', ',') # 解码器: 整数列表 -> 字符串 (简单的后处理)

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
best_val_loss = float('inf') # 记录最佳验证集损失
pbar = tqdm(range(max_iters), desc="Training")
for iter in pbar:

    # 每隔一段时间评估一次损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        pbar.set_postfix({'train_loss': f"{losses['train']:.4f}", 'val_loss': f"{losses['val']:.4f}"})
        
        # 如果验证集损失更低，保存模型
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
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
            # print(f"New best model saved with val_loss: {best_val_loss:.4f}") # 可选：打印提示

    # 采样一批数据
    xb, yb = get_batch('train')

    # 评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete.")
print(f"Best validation loss was: {best_val_loss:.4f}")

# --- 保存模型 ---
# print("Saving model...")
# torch.save(...) # 已经移到循环内部，只保存最佳模型
print("Model saved to gpt_model.pth (Best version)")

# --- 生成示例 ---
print("Generating sample text:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
