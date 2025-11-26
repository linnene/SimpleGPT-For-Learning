# SimpleGPT 深度解读文档

本文档旨在帮助初学者理解 `model.py` 中实现的 GPT 模型架构。我们将从底层组件开始，逐步构建出完整的 GPT。

## 1. 核心概念：Transformer 与 GPT

GPT (Generative Pre-trained Transformer) 是基于 **Transformer** 架构的 **Decoder-only**（仅解码器）模型。
*   **Transformer**: 一种基于“注意力机制”的深度学习模型，擅长处理序列数据（如文本）。
*   **Decoder-only**: GPT 只需要“生成”下一个字，所以它只用了 Transformer 的解码部分（去掉了编码器 Encoder 和解码器中的 Encoder-Decoder Attention）。

## 2. 代码组件详解

### 2.1 `Head` (自注意力头)

这是模型最核心的部分。它的作用是让序列中的每个 token（字符）都能“看到”并聚合它前面的 token 的信息。

*   **Query (Q), Key (K), Value (V)**: 
    *   每个 token 发出一个 Query（我在找什么？）。
    *   每个 token 发出一个 Key（我是什么？）。
    *   每个 token 发出一个 Value（如果我被选中，我包含什么信息？）。
*   **Attention Score (注意力分数)**: `wei = q @ k`。通过 Query 和 Key 的点积计算相关性。如果 Q 和 K 相似，分数就高。
*   **Masking (掩码)**: `tril` 矩阵用于确保模型**只能看到过去，不能看到未来**。这是 GPT 生成式模型的关键。我们在计算出的注意力分数矩阵上，把“未来”的位置设为负无穷大，这样 Softmax 后概率为 0。
*   **Aggregation (聚合)**: `out = wei @ v`。根据注意力分数，加权求和 Value。

### 2.2 `MultiHeadAttention` (多头注意力)

一个 Head 可能只能关注到一种关系（例如：主语和谓语的关系）。为了捕捉更多样的特征，我们并行运行多个 Head，然后把结果拼接起来。

*   就像让几个人同时读一句话，每个人关注不同的侧重点，最后汇总意见。

### 2.3 `FeedFoward` (前馈神经网络)

注意力机制负责“收集信息”（通信），而前馈网络负责“思考和处理信息”（计算）。
*   它是一个简单的多层感知机 (MLP)，包含线性层和非线性激活函数 (ReLU)。
*   它独立地作用于每个 token。

### 2.4 `Block` (Transformer 块)

一个标准的 Transformer 块由以下部分组成：
1.  **LayerNorm (层归一化)**: 稳定训练。
2.  **MultiHeadAttention**: 通信。
3.  **FeedForward**: 计算。
4.  **Residual Connection (残差连接)**: `x = x + ...`。这允许梯度直接流过网络，解决了深层网络难以训练的问题（梯度消失）。

### 2.5 `GPTLanguageModel` (整体架构)

这是最终的模型类。

1.  **Embedding (嵌入层)**:
    *   `token_embedding_table`: 把每个字符 ID 转换成一个向量。
    *   `position_embedding_table`: 告诉模型每个字符在句子中的位置（因为 Attention 机制本身不包含位置信息）。
2.  **Blocks**: 堆叠多个 Transformer Block。层数越深，模型越强。
3.  **Head (输出层)**: 最后的线性层，把向量映射回词表大小，输出每个字符的概率 logits。

## 3. 训练过程 (`train.py`)

1.  **数据预处理**: 把文本中的字符转换成整数 ID (Tokenization)。
2.  **Batching**: 随机抽取一小段文本（例如长度 64），作为输入 `x`。目标 `y` 是 `x` 向后移一位的序列。
    *   输入: "Hell"
    *   目标: "ello"
    *   模型需要学会：输入 "H" -> 预测 "e"; 输入 "He" -> 预测 "l"; ...
3.  **Loss (损失函数)**: 使用 `CrossEntropyLoss` 衡量预测概率分布与真实字符的差异。
4.  **Optimization**: 使用 `AdamW` 优化器更新权重。

## 4. 总结

这个简单的 GPT 虽然小，但它包含了现代大语言模型（如 GPT-3, GPT-4）的所有核心数学原理。区别仅在于：
*   **规模**: 它们有更多的层、更多的头、更大的嵌入维度。
*   **数据**: 它们使用了海量的互联网文本。
*   **Tokenizer**: 它们使用 BPE (Byte Pair Encoding) 而不是简单的字符级映射。
