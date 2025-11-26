# 使用官方的 PyTorch 镜像作为基础，包含 CUDA 支持
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
# (这样做的好处是：如果代码变了但依赖没变，Docker 会利用缓存，不用重新安装包)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码到容器中
COPY . .

# 默认运行训练脚本
CMD ["python", "train.py"]
