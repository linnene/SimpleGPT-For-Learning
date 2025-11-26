import torch
from model import GPTLanguageModel
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 加载模型 ---
if not os.path.exists('gpt_model.pth'):
    print("Error: gpt_model.pth not found. Please run train.py first.")
    exit()

checkpoint = torch.load('gpt_model.pth', map_location=device)

stoi = checkpoint['stoi']
itos = checkpoint['itos']
block_size = checkpoint['block_size']
n_embd = checkpoint['n_embd']
n_head = checkpoint['n_head']
n_layer = checkpoint['n_layer']
vocab_size = checkpoint['vocab_size']

model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout=0.0, device=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 交互式生成 ---
print("\nModel loaded successfully!")
print("Type a prompt and press Enter to generate text (or 'quit' to exit).")

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() == 'quit':
        break
    
    if not prompt:
        prompt = " " # 空提示

    try:
        # 编码输入
        context_idxs = encode(prompt)
        context = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

        # 生成
        print("Generating...", end='\r')
        generated_idxs = model.generate(context, max_new_tokens=500)
        generated_text = decode(generated_idxs[0].tolist())

        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------")
    except KeyError as e:
        print(f"\nError: Your prompt contains characters not in the training vocabulary: {e}")
