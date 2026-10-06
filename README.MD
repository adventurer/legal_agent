# 1. 创建并激活 Python 3.10 隔离环境
conda create -n legal_agent python=3.10 -y
conda activate legal_agent

# 2. 安装 CUDA 兼容的 PyTorch (以 CUDA 12.1 为例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 vLLM 推理引擎、ModelScope 下载工具及 HTTP 客户端
pip install vllm==0.6.3 modelscope requests
pip install --no-cache-dir vllm==0.6.4.post1 modelscope requests "transformers>=4.45.0,<4.47.0"

# pdf支持
pip install reportlab
pip install pymupdf

# 光学支持
pip install rapidocr-onnxruntime pillow
pip install pytesseract 可选
安装字体以适配图片格式自检
sudo apt update && sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei

# Word 文档支持（.docx 直接解析；Linux/WSL 下 .doc 使用 LibreOffice 转换）
pip install python-docx
# Linux/WSL 读取旧版 .doc
sudo apt update && sudo apt install -y libreoffice
# Windows 读取旧版 .doc 的可选方案
# pip install pywin32

# gateway
pip install fastapi uvicorn sse-starlette python-multipart

# webui
pip install streamlit httpx-sse

# 审查意见导出
# Streamlit 支持下载 Markdown (.md) 和 Word (.docx) 两种格式

# 系统提供合同审查和审查意见导出，不提供自动生成或导出修订合同的功能。

# 审查过程不持久化保存合同正文、模型输入输出或工具交互记录。

# 4. 创建项目专属目录并进入
mkdir -p ~/legal_agent_lab && cd ~/legal_agent_lab

# 5.检查
python -c "import torch, vllm; print('>>> 环境完全正常！CUDA可用:', torch.cuda.is_available(), '| PyTorch:', torch.__version__, '| vLLM:', vllm.__version__)"

# 6.查模型
curl -i http://127.0.0.1:8000/v1/models


# useage:
python server/vllm_launcher.py --model qwen2.5-7b-awq --no-fp8-kv
python server/vllm_launcher.py --model qwen2.5-3b --no-fp8-kv
# Qwen3.5-4B uses an AWQ INT4 text-only checkpoint, Qwen3 tool parsers, and a 12K context; first launch downloads it.
python server/vllm_launcher.py --model qwen3.5-4b --no-fp8-kv
python gateway/api_server.py
streamlit run web/main.py --server.port 8501

# 可选运行配置
# MAX_UPLOAD_BYTES=20971520      # 合同上传大小上限，默认 20 MiB
# VLLM_MAX_MODEL_LEN=4096        # 可选：覆盖当前模型的默认上下文长度
# DEFAULT_MODEL_NAME=qwen2.5-7b  # 默认模型；权重目录、启动别名及上下文长度由共享模型注册表自动解析
# DEFAULT_MODEL_KEY=qwen2.5-7b-awq  # 可选：使用启动器模型键覆盖默认模型

# 模块职责
# services/doc_loader.py 仅负责多格式文档读取与文本归一化
# services/clause_splitter.py 负责合同条款切分
# services/document_models.py 定义文档领域模型
# gateway/upload_service.py 负责上传校验与临时文件写入
# web/api_client.py 负责 Streamlit 到 FastAPI 网关的 HTTP 调用
# archive/ 保存已确认不再作为运行入口的历史副本
