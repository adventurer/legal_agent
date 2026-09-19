#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: configs/config.py
职责:
1. 统一管理全系统核心路径 (ROOT_DIR, reference_docs, models, data)
2. 统一管理推理服务 (vLLM) 与 API 网关 (FastAPI) 的监听地址与端口
3. 统一管理模型推断超参数 (seed, max_tokens, temperature) 与超时配置
4. 支持通过环境变量动态覆盖配置，并在初始化时自动创建必备目录
"""

import os
from pathlib import Path
from typing import Dict, Any


# ==================== 1. 核心目录与路径配置 ====================
# 获取项目根目录 (configs/ 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型权重存放目录
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))

# 权威参考法规与合规 PDF 目录
REFERENCE_DOCS_DIR = Path(os.getenv("REFERENCE_DOCS_DIR", PROJECT_ROOT / "reference_docs"))

# 系统数据持久化目录 (用于 SQLite 数据库、上传临时文件等)
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

# 企业自编法典 SQLite 数据库文件路径
RULE_BOOK_DB_PATH = DATA_DIR / "rule_book.db"

# 自动确保必要目录存在
for directory in [REFERENCE_DOCS_DIR, DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================== 2. 模型与推理服务 (vLLM) 配置 ====================
# 默认部署的模型名称与物理子目录
DEFAULT_MODEL_NAME = "qwen2.5-7b"
DEFAULT_MODEL_DIR_NAME = "Qwen2.5-7B-Instruct-AWQ"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_DIR_NAME

# vLLM 服务端监听配置
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))

# 客户端连接基础地址 (强制走 127.0.0.1 避免 DNS 与代理问题)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_PORT}/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none")

# vLLM 启动与显存控制参数
VLLM_CONFIG: Dict[str, Any] = {
    "model_path": str(DEFAULT_MODEL_PATH),
    "served_name": DEFAULT_MODEL_NAME,
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEM_UTIL", "0.85")),
    "quantization": "awq",
    "kv_cache_dtype": "auto",  # 保持原生 FP16，严禁在此开启 FP8
}


# ==================== 3. 核心 Agent 推理超参数 ====================
AGENT_CONFIG: Dict[str, Any] = {
    "model": DEFAULT_MODEL_NAME,
    "temperature": 0.0,       # 设为 0 启用贪婪解码，保持法律审查结论高度确定
    "top_p": 1.0,
    "seed": 42,               # 显式固定随机种子
    "max_tokens": 1024,       # 单步生成最大 Token 数
    "stop": ["Observation:"], # ReAct 工具调用截断符
    "max_turns": 6,           # 默认最大推理轮次
}

# 网络与客户端超时参数 (秒)
TIMEOUT_CONFIG = {
    "total_timeout": float(os.getenv("CLIENT_TIMEOUT", "60.0")),
    "read_timeout": float(os.getenv("CLIENT_READ_TIMEOUT", "60.0")),
    "connect_timeout": 10.0,
    "max_retries": 2,
}


# ==================== 4. API 网关与前端 UI 配置 ====================
# FastAPI 网关配置
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "9000"))

# Streamlit 前端服务端口
WEB_PORT = int(os.getenv("WEB_PORT", "8501"))


# ==================== 本地快速调试打印 ====================
if __name__ == "__main__":
    print("=" * 60)
    print(" Legal Agent Lab 系统全局配置一览")
    print("=" * 60)
    print(f"项目根路径 (PROJECT_ROOT)    : {PROJECT_ROOT}")
    print(f"参考文档目录 (REFERENCE_DOCS): {REFERENCE_DOCS_DIR}")
    print(f"自编法典数据库 (RULE_BOOK_DB): {RULE_BOOK_DB_PATH}")
    print(f"模型权重路径 (MODEL_PATH)    : {DEFAULT_MODEL_PATH}")
    print(f"vLLM 服务端 URL              : {VLLM_BASE_URL}")
    print(f"FastAPI 网关监听             : {GATEWAY_HOST}:{GATEWAY_PORT}")
    print(f"Streamlit 前端端口           : {WEB_PORT}")
    print("=" * 60)
    print("目录存在性检查:")
    print(f" - reference_docs 存在: {REFERENCE_DOCS_DIR.exists()}")
    print(f" - data 存在          : {DATA_DIR.exists()}")
    print(f" - 模型权重目录存在   : {DEFAULT_MODEL_PATH.exists()}")
    print("=" * 60)