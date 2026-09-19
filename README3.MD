## 🛠️ 环境准备与安装

依次执行以下步骤完成系统字体、虚拟环境、全量依赖安装与资源放置：

### 1. 系统配置与一键安装依赖
在 Linux / WSL2 终端中依次运行系统字体安装、创建并激活 Conda 环境，随后一次性安装全量 Python 依赖包：

```bash
# 1. 安装系统级中文字体（确保合同图片 OCR 与渲染正常）
sudo apt update && sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei

# 2. 创建并激活 Python 3.10 虚拟环境
conda create -n legal_agent python=3.10 -y
conda activate legal_agent

# 3. 安装项目所需全部 Python 核心依赖库
pip install \
    vllm \
    openai \
    httpx \
    httpx-sse \
    pydantic \
    pypdf \
    python-docx \
    rapidocr-onnxruntime \
    pillow \
    fastapi \
    uvicorn \
    sse-starlette \
    python-multipart \
    streamlit

# 4. 模型权重与参考文档配置
# 创建目录（如未自动生成）
mkdir -p models reference_docs data

# 1. 模型权重：放入 Qwen2.5-7B-Instruct-AWQ 权重文件
# 路径目标: legal_agent_lab/models/Qwen2.5-7B-Instruct-AWQ/

# 2. 权威文档：放入参考民法典及合规政策 PDF 文件
# 路径目标: legal_agent_lab/reference_docs/*.pdf