#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/api_server.py
职责:
1. 暴露 RESTful 接口供 Web 前端调用
2. 提供合同文件（Word / PDF / 图片 OCR / TXT）上传与条款切分接口
3. 提供基于 SSE 的打字机式流式推理审查接口
4. 核心健壮性与可观测性：
   - 任务级端到端追踪：日志全程注入 Task ID 前缀
   - 熔断事件透传：新增 circuit_break SSE 事件，实时向前端推送熔断详情与知识库补齐建议
   - 纯输入 Token 95% 熔断底线：输入接近极限时自动切入模型自知识推理
   - 全程 4096 预算，末轮解除 stop 词杜绝长文截断
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import re
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# 项目根目录对齐
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    DATA_DIR,
    AGENT_CONFIG,
    MAX_UPLOAD_BYTES,
)
from core.schemas import ReviewRequest, AgentExecutionResult
from core.agent_loop import ContractReviewAgent
from services.doc_loader import DocumentLoader
from gateway.session_manager import session_manager
from gateway.upload_service import create_temp_upload_path, get_safe_extension, save_upload_file
from gateway.review_orchestrator import (
    CONTEXT_THRESHOLD_95,
    MODEL_MAX_CONTEXT,
    OUTPUT_MAX_TOKENS,
    ReviewOrchestrator,
    generate_kb_deficit_recommendation,
)

app = FastAPI(
    title="Legal Agent Lab API Gateway",
    description="本地法务合同审查智能体高并发网关服务",
    version="1.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance = ContractReviewAgent()
review_orchestrator = ReviewOrchestrator(agent_instance)
doc_loader = DocumentLoader()


def normalize_clauses_output(raw_clauses: Any) -> List[Dict[str, Any]]:
    formatted = []
    if not raw_clauses:
        return formatted
    for i, c in enumerate(raw_clauses):
        if hasattr(c, "to_dict") and callable(c.to_dict):
            formatted.append(c.to_dict())
        elif isinstance(c, dict):
            formatted.append({
                "index": c.get("index", i + 1),
                "title": c.get("title", f"条款 {i + 1}"),
                "content": c.get("content", ""),
                "type": c.get("type", "article"),
            })
        else:
            formatted.append({
                "index": getattr(c, "index", i + 1),
                "title": getattr(c, "title", f"条款 {i + 1}"),
                "content": getattr(c, "content", str(c)),
                "type": getattr(c, "type", "article"),
            })
    return formatted


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": agent_instance.model_name,
        "tools_ready": list(agent_instance.tool_mapping.keys()),
        "context_threshold_95": CONTEXT_THRESHOLD_95,
    }


@app.post("/api/v1/sessions/create")
async def create_session():
    session_id = session_manager.create_session()
    return {"code": 200, "message": "success", "data": {"session_id": session_id}}


@app.post("/api/v1/contract/upload")
async def upload_contract(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    if not session_id or not session_manager.get_session(session_id):
        session_id = session_manager.create_session()

    extension = get_safe_extension(file.filename or "")
    upload_dir = DATA_DIR / "uploads"
    temp_file_path = create_temp_upload_path(upload_dir, session_id, extension)

    try:
        await save_upload_file(file, temp_file_path, MAX_UPLOAD_BYTES)

        raw_clauses = doc_loader.load_and_split(temp_file_path)
        clauses_data = normalize_clauses_output(raw_clauses)
        session_manager.store_clauses(session_id, clauses_data)

        return {
            "code": 200,
            "message": "success",
            "data": {
                "session_id": session_id,
                "filename": file.filename,
                "clause_count": len(clauses_data),
                "clauses": clauses_data,
            },
        }
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError, ImportError) as e:
        raise HTTPException(status_code=422, detail=f"文件处理失败: {e}") from e
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@app.post("/api/v1/contract/review/stream")
async def review_contract_stream(request: ReviewRequest):
    limit_turns = request.max_turns or AGENT_CONFIG.get("max_turns", 6)
    clause_title_match = re.search(
        r"^(第[一二三四五六七八九十百0-9]+条[^\n]+|合同前言[^\n]*)",
        request.contract_text.strip(),
    )
    task_label = clause_title_match.group(1).strip() if clause_title_match else f"Task-{uuid.uuid4().hex[:6]}"
    return EventSourceResponse(
        review_orchestrator.stream(request.contract_text, limit_turns, task_label)
    )


def main():
    parser = argparse.ArgumentParser(description="启动 Legal Agent FastAPI Gateway")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="在控制台打印每轮模型请求、响应和工具交互详情",
    )
    args = parser.parse_args()
    review_orchestrator.debug = args.debug
    print(f"[*] 正在启动 Legal Agent FastAPI Gateway 监听: http://{GATEWAY_HOST}:{GATEWAY_PORT}")
    print(f"[*] 模型交互 Debug: {'开启' if args.debug else '关闭'}")
    uvicorn.run(
        app,
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()