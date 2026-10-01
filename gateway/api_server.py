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
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import re
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
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
from core.schemas import (
    ReviewRequest,
    AgentExecutionResult,
    ContractRewriteRequest,
    ContractRewriteResponse,
)
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
from services.contract_rewriter import rewrite_selected_clauses
from services.review_trace import ReviewTrace
from gateway.model_runtime import model_runtime

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
                "raw_text": c.get("raw_text", ""),
                "type": c.get("type", "article"),
            })
        else:
            formatted.append({
                "index": getattr(c, "index", i + 1),
                "title": getattr(c, "title", f"条款 {i + 1}"),
                "content": getattr(c, "content", str(c)),
                "raw_text": getattr(c, "raw_text", ""),
                "type": getattr(c, "type", "article"),
            })
    return formatted


@app.get("/health")
async def health_check():
    from gateway import review_orchestrator as orchestrator_module

    runtime_status = model_runtime.status()
    unavailable_tools = getattr(agent_instance.tool_registry, "unavailable_tools", set())
    return {
        "status": "healthy",
        "model": agent_instance.model_name,
        "inference_model": runtime_status.get("model"),
        "model_matches_inference": runtime_status.get("model") == agent_instance.model_name,
        "tools_ready": [
            name for name in agent_instance.tool_mapping if name not in unavailable_tools
        ],
        "tools_unavailable": sorted(unavailable_tools),
        "knowledge_base_pages": len(
            getattr(agent_instance.tool_registry.knowledge_base, "pages", []) or []
        ),
        "context_threshold_95": orchestrator_module.CONTEXT_THRESHOLD_95,
    }


@app.get("/api/v1/model-runtime")
async def get_model_runtime():
    return model_runtime.status()


@app.get("/api/v1/models")
async def get_supported_models():
    from configs.config import MODEL_PRESETS

    return {
        "models": [
            {"key": key, "name": preset["served_name"], "context": preset["max_model_len"]}
            for key, preset in MODEL_PRESETS.items()
        ]
    }


@app.post("/api/v1/model-runtime/start")
async def start_model_runtime(payload: Dict[str, str] = Body(...)):
    try:
        return model_runtime.start(payload.get("model_key", ""))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"无法启动模型服务：{exc}") from exc


@app.post("/api/v1/model-runtime/stop")
async def stop_model_runtime():
    try:
        return model_runtime.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    # A vLLM process may have been started from a terminal rather than the page.
    # Reconcile its served name before constructing any model requests.
    runtime_status = model_runtime.status()
    inference_model = runtime_status.get("model")
    if not inference_model:
        raise HTTPException(status_code=503, detail="vLLM 推理服务不可用，请先启动模型服务。")
    if inference_model != agent_instance.model_name:
        raise HTTPException(
            status_code=503,
            detail=(
                f"网关模型同步失败：网关配置为 {agent_instance.model_name}，"
                f"但 vLLM 当前提供 {inference_model}。请刷新网关服务状态或重启网关。"
            ),
        )
    limit_turns = request.max_turns or AGENT_CONFIG.get("max_turns", 6)
    clause_title_match = re.search(
        r"^(第[一二三四五六七八九十百0-9]+条[^\n]+|合同前言[^\n]*)",
        request.contract_text.strip(),
    )
    task_label = clause_title_match.group(1).strip() if clause_title_match else f"Task-{uuid.uuid4().hex[:6]}"
    review_run_id = request.review_run_id or uuid.uuid4().hex

    try:
        trace = ReviewTrace(
            DATA_DIR / "review_logs",
            contract_text=request.contract_text,
            max_turns=limit_turns,
            model=agent_instance.model_name,
            task_label=task_label,
            review_run_id=review_run_id,
        )
        print(f"[*] 审查调试记录: {trace.path}", flush=True)
    except OSError as exc:
        print(f"[警告] 无法创建审查调试记录: {exc}", flush=True)
        trace = None

    async def traced_stream():
        trace_status = "interrupted"
        try:
            async for item in review_orchestrator.stream(
                request.contract_text,
                limit_turns,
                task_label,
                request.review_side,
            ):
                if item.get("event") == "start" and trace:
                    try:
                        payload = json.loads(item.get("data", "{}"))
                        payload["trace_id"] = trace.trace_id
                        item = {**item, "data": json.dumps(payload, ensure_ascii=False)}
                    except (TypeError, ValueError):
                        pass
                if trace:
                    trace.record_sse(item)
                if item.get("event") == "final_report":
                    try:
                        result = json.loads(item.get("data", "{}"))
                        trace_status = result.get("status", "completed") if result.get("is_complete") else "incomplete"
                    except (TypeError, ValueError):
                        trace_status = "incomplete"
                elif item.get("event") == "error":
                    trace_status = "error"
                elif item.get("event") == "done" and trace_status == "interrupted":
                    trace_status = "finished_without_final_report"
                yield item
        except Exception as exc:
            trace_status = "error"
            if trace:
                trace.record_sse({
                    "event": "gateway_exception",
                    "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
                })
            raise
        finally:
            if trace:
                trace.close(trace_status)

    return EventSourceResponse(
        traced_stream()
    )


@app.post("/api/v1/contract/rewrite", response_model=ContractRewriteResponse)
async def rewrite_contract(request: ContractRewriteRequest):
    try:
        result = rewrite_selected_clauses(
            agent_instance,
            request.clauses,
            request.review_report,
            request.selected_indices,
        )
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
