# SimpleGPT 项目开发文档

这是一个用于学习和演示目的的简单 GPT (Generative Pre-trained Transformer) 实现。它包含了一个基于 PyTorch 的微型 GPT 模型，可以在简单的文本数据集（如莎士比亚作品集）上进行训练和生成。

## 1. 环境准备

确保你已经安装了 Python 3.8+。

安装依赖项：

```bash
pip install -r requirements.txt
```

*注意：如果你有 NVIDIA 显卡，建议安装支持 CUDA 的 PyTorch 版本以加速训练。*

## 2. 项目结构

*   `model.py`: 定义了 GPT 模型的架构，包括 Self-Attention, FeedForward, Block 等组件。
*   `train.py`: 负责下载数据、预处理数据、训练模型并保存模型权重。
*   `generate.py`: 加载训练好的模型，并根据用户输入的提示词生成文本。
*   `requirements.txt`: 项目依赖列表。

## 3. 运行训练

运行 `train.py` 脚本开始训练模型。该脚本会自动下载 `tiny_shakespeare` 数据集（如果下载失败会创建一个简单的测试数据集）。

```bash
python train.py
```

**训练过程说明：**
*   脚本会打印出训练集和验证集的 Loss（损失值）。Loss 越低，模型效果越好。
*   训练结束后，模型权重和配置会保存到 `gpt_model.pth` 文件中。
*   脚本最后会生成一段示例文本。

**超参数调整：**
你可以在 `train.py` 顶部修改超参数，例如：
*   `max_iters`: 训练迭代次数（默认 2000，增加可提高效果但耗时更长）。
*   `batch_size`: 批次大小。
*   `n_layer` / `n_head` / `n_embd`: 模型的大小配置。

## 4. 文本生成

训练完成后，运行 `generate.py` 来体验模型生成。

```bash
python generate.py
```

输入一段提示词（Prompt），模型将尝试续写。

## 5. 常见问题

*   **报错 `KeyError`**: 如果你在提示词中输入了训练数据中从未出现过的字符（例如中文字符，而训练集全是英文），模型会报错。这是因为这是一个简单的字符级 GPT，词表仅包含训练集中出现的字符。
*   **生成结果乱码或不通顺**: 
    *   训练时间太短（增加 `max_iters`）。
    *   模型太小（增加 `n_layer`, `n_embd`）。
    *   这是字符级模型，相比词级模型（如 ChatGPT 使用的 BPE），它需要学习单词拼写，难度较大。
