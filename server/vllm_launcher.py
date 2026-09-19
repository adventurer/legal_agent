#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 推理服务启动器
职责：
1. 统一管理模型目录（严格存放于 models/ 目录）、别名映射与 ModelScope 自动拉取
2. 注入 WSL2 防死锁环境变量 (TORCH_UVA_ENABLE=0)
3. 针对 8G~16G 显存默认启用 KV Cache FP8 量化与 eager 模式
4. 支持 CPU Offload 解决显存边缘溢出问题
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.config import DEFAULT_MODEL_KEY

# 获取项目根目录 (假设当前脚本位于 legal_agent_lab/server/)
# 统一且唯一的模型存放基目录：legal_agent_lab/models/
MODELS_BASE_DIR = ROOT_DIR / "models"

# ==================== 支持的模型配置注册表 ====================
SUPPORTED_MODELS = {
    # 3B 原生指令版（超轻量，极低显存设备兜底）
    "qwen2.5-3b": {
        "modelscope_id": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "model_dir": MODELS_BASE_DIR / "Qwen2.5-3B-Instruct-AWQ",
        "quantization": "awq",
        "max_model_len": 4096,
        "gpu_utilization": 0.80,
        "served_name": "qwen2.5-3b",
    },
    # 7B AWQ 量化版（8GB 显存主力推荐）
    "qwen2.5-7b-awq": {
        "modelscope_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "model_dir": MODELS_BASE_DIR / "Qwen2.5-7B-Instruct-AWQ",
        "quantization": "awq",
        "max_model_len": 8192,
        "gpu_utilization": 0.85,
        "served_name": "qwen2.5-7b",
    },
    # 14B AWQ 量化版（适合 12G/16G 显存）
    "qwen2.5-14b-awq": {
        "modelscope_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "model_dir": MODELS_BASE_DIR / "Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq",
        "max_model_len": 4096,
        "gpu_utilization": 0.88,
        "served_name": "qwen2.5-14b",
    },
    # DeepSeek 蒸馏 7B AWQ
    "deepseek-r1-7b-awq": {
        "modelscope_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "model_dir": MODELS_BASE_DIR / "DeepSeek-R1-Distill-Qwen-7B-AWQ",
        "quantization": "awq",
        "max_model_len": 4096,
        "gpu_utilization": 0.85,
        "served_name": "deepseek-r1-7b",
    },
}


def is_valid_model_dir(path: Path) -> bool:
    """检查目录下是否包含完整的配置文件与模型权重"""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_weights


def ensure_model_weights(config: dict) -> Path:
    """检查 models/ 目录下的模型权重是否存在，若不存在则自动拉取"""
    model_path = Path(config["model_dir"]).resolve()

    if is_valid_model_dir(model_path):
        print(f"[*] 检测到模型权重完整: {model_path}")
        return model_path

    print(f"[!] 本地未找到完整权重，准备从 ModelScope 下载至: {model_path}")
    try:
        from modelscope import snapshot_download

        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_path = snapshot_download(
            model_id=config["modelscope_id"],
            local_dir=str(model_path),
        )
        print(f"[+] 下载完成，权重已保存至: {download_path}")
        return Path(download_path)
    except ImportError:
        print("[-] 错误: 未安装 modelscope。请执行 `pip install modelscope`。")
        sys.exit(1)
    except Exception as e:
        print(f"[-] 下载模型失败: {e}")
        sys.exit(1)


def launch_vllm_server(
    model_key: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    gpu_util: float = None,
    max_len: int = None,
    cpu_offload: int = 0,
    enable_fp8_kv: bool = True,
    seed: int = 42,
):
    """构建参数并启动 vLLM OpenAI API 服务进程"""
    if model_key not in SUPPORTED_MODELS:
        print(f"[-] 不支持的模型标识: {model_key}")
        print(f"[*] 可选配置列表: {list(SUPPORTED_MODELS.keys())}")
        sys.exit(1)

    cfg = SUPPORTED_MODELS[model_key]
    model_path = ensure_model_weights(cfg)

    actual_gpu_util = gpu_util if gpu_util is not None else cfg["gpu_utilization"]
    actual_max_len = max_len if max_len is not None else cfg["max_model_len"]

    # 1. 注入 WSL2 必要的底层环境变量
    env = os.environ.copy()
    env["TORCH_UVA_ENABLE"] = "0"  # 禁用 UVA 避免 WSL2 共享显存死锁
    env["PYTHONUNBUFFERED"] = "1"

    # 2. 拼接 vllm 启动参数
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        cfg["served_name"],
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(actual_max_len),
        "--gpu-memory-utilization",
        str(actual_gpu_util),
        "--seed",
        str(seed),
        "--enforce-eager",
    ]

    if cfg.get("quantization"):
        cmd.extend(["--quantization", cfg["quantization"]])

    if enable_fp8_kv:
        cmd.extend(["--kv-cache-dtype", "fp8"])

    if cpu_offload > 0:
        cmd.extend(["--cpu-offload-gb", str(cpu_offload)])

    print("=" * 65)
    print(" 🚀 正在拉起 vLLM 工业级推理服务")
    print("=" * 65)
    print(f" 模型标识: {model_key}")
    print(f" 权重路径: {model_path}")
    print(f" 对外暴露名: {cfg['served_name']}")
    print(f" 监听地址: http://{host}:{port}/v1")
    print(f" 上下文上限: {actual_max_len} tokens")
    print(f" 显存利用率: {actual_gpu_util * 100:.0f}%")
    print(f" KV Cache: {'FP8 量化' if enable_fp8_kv else '原生 FP16'}")
    if cpu_offload > 0:
        print(f" 内存卸载: {cpu_offload} GB (已挂载到系统内存)")
    print(" 完整执行命令:")
    print(" " + " ".join(cmd))
    print("=" * 65)

    try:
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    except KeyboardInterrupt:
        print("\n[*] 正在关闭 vLLM 推理服务...")
        process.terminate()
        process.wait()
        print("[+] 服务已安全退出。")


def main():
    parser = argparse.ArgumentParser(description="Legal Agent Lab - vLLM 服务管理脚本")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL_KEY,
        choices=list(SUPPORTED_MODELS.keys()),
        help="待启动的模型标识 (默认: qwen2.5-7b-awq)",
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="服务监听端口 (默认: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听网段 (默认: 0.0.0.0)")
    parser.add_argument("--gpu-util", type=float, default=None, help="覆盖默认 GPU 显存利用率 (如 0.85)")
    parser.add_argument("--max-len", type=int, default=None, help="覆盖默认最大模型上下文序列长度 (如 4096)")
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=0,
        help="卸载到主机系统内存(RAM)的模型大小(单位: GB，默认: 0)",
    )
    parser.add_argument(
        "--no-fp8-kv",
        action="store_true",
        help="禁用 KV Cache FP8 压缩，回退到 FP16",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机采样基准种子 (默认: 42)")

    args = parser.parse_args()

    launch_vllm_server(
        model_key=args.model,
        port=args.port,
        host=args.host,
        gpu_util=args.gpu_util,
        max_len=args.max_len,
        cpu_offload=args.cpu_offload,
        enable_fp8_kv=not args.no_fp8_kv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()