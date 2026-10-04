#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/api_server.py
职责:
1. 暴露 RESTful 接口供 Web 前端调用
2. 提供合同文件（Word / PDF / 图片 OCR / TXT）上传与条款切分接口
3. 提供基于 SSE 的打字机式流式推理审查接口
4. 保留审查轨迹记录与 ReAct 流式事件
"""

import os
import sys
import asyncio
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Any, List

import re
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from urllib.parse import urlparse

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
)
from core.agent_loop import ContractReviewAgent
from services.doc_loader import DocumentLoader
from gateway.session_manager import session_manager
from gateway.upload_service import create_temp_upload_path, get_safe_extension, save_upload_file
from gateway.review_orchestrator import ReviewOrchestrator
from gateway.model_runtime import model_runtime
from services.pdf_kb_loader import (
    convert_law_source_to_txt,
    determine_tag,
    law_text_related_files,
    pdf_text_cache_path,
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

KNOWLEDGE_FILE_EXTENSIONS = {".pdf", ".txt", ".md"}
KNOWLEDGE_CATEGORY_PREFIXES = {
    "law": "法典",
    "enterprise": "合规",
    "general": "通用",
}


def _knowledge_source_paths() -> List[Path]:
    docs_dir = Path(agent_instance.docs_dir)
    if not docs_dir.exists():
        return []
    return sorted(
        path for path in docs_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in KNOWLEDGE_FILE_EXTENSIONS
        and not path.name.lower().endswith((".pdf.txt", ".pdf.law.txt"))
        and (
            path.name.lower().endswith(".law.txt")
            or determine_tag(path.name) != "法"
        )
    )


def _reload_knowledge_index() -> int:
    agent_instance.tool_mapping = agent_instance.tool_registry.build()
    knowledge_base = agent_instance.tool_registry.knowledge_base
    return len(getattr(knowledge_base, "pages", []) or [])


def _editable_knowledge_source(filename: str, extensions: set[str]) -> Path:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="无效的文件名。")
    docs_dir = Path(agent_instance.docs_dir).resolve()
    source = docs_dir / filename
    if (
        not source.is_file()
        or source.is_symlink()
        or source.suffix.lower() not in extensions
        or source.name.lower().endswith((".pdf.txt", ".pdf.law.txt"))
    ):
        raise HTTPException(status_code=404, detail="未找到可编辑的知识库资料。")
    if determine_tag(source.name) not in {"合规", "通用"}:
        raise HTTPException(status_code=403, detail="仅允许编辑企业知识库和其他资料。")
    return source


async def _replace_knowledge_source(source: Path, content: bytes) -> int:
    artifacts = [source.with_suffix(".articles.json")]
    if source.suffix.lower() == ".pdf":
        for pdf_text in {pdf_text_cache_path(source), source.with_suffix(".pdf.txt")}:
            artifacts.extend([pdf_text, pdf_text.with_suffix(pdf_text.suffix + ".meta.json")])
    previous_artifacts = {
        artifact: artifact.read_bytes()
        for artifact in artifacts
        if artifact.is_file() and not artifact.is_symlink()
    }
    previous_content = source.read_bytes()
    temporary_path = source.parent / f".kb-edit-{uuid.uuid4().hex}.tmp"
    replaced = False
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, source)
        replaced = True
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        return await asyncio.to_thread(_reload_knowledge_index)
    except (OSError, RuntimeError, ValueError) as exc:
        if replaced:
            rollback_path = source.parent / f".kb-rollback-{uuid.uuid4().hex}.tmp"
            try:
                rollback_path.write_bytes(previous_content)
                os.replace(rollback_path, source)
                for artifact in artifacts:
                    if artifact in previous_artifacts:
                        artifact.write_bytes(previous_artifacts[artifact])
                    else:
                        artifact.unlink(missing_ok=True)
                await asyncio.to_thread(_reload_knowledge_index)
            except (OSError, RuntimeError, ValueError):
                pass
            finally:
                rollback_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"资料保存或索引失败：{exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _knowledge_rule_book():
    rule_book = agent_instance.tool_registry.rule_book
    if rule_book is None:
        raise HTTPException(status_code=503, detail="自编法典服务当前不可用。")
    return rule_book


def _require_local_knowledge_admin(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    origin = request.headers.get("origin")
    origin_host = urlparse(origin).hostname if origin else None
    if client_host not in {"127.0.0.1", "::1", "testclient"} or (
        origin_host and origin_host not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise HTTPException(status_code=403, detail="知识库管理接口仅允许本机访问。")


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
    }


@app.get("/api/v1/knowledge/documents")
def list_knowledge_documents(request: Request):
    _require_local_knowledge_admin(request)
    knowledge_base = agent_instance.tool_registry.knowledge_base
    pages = getattr(knowledge_base, "pages", []) or []
    indexed_counts = Counter(page.get("doc_name") for page in pages)
    category_by_tag = {"法": "law", "合规": "enterprise", "通用": "general"}
    documents = []
    for path in _knowledge_source_paths():
        tag = determine_tag(path.name)
        documents.append({
            "filename": path.name,
            "category": category_by_tag.get(tag, "general"),
            "tag": tag,
            "size_bytes": path.stat().st_size,
            "indexed_segments": indexed_counts.get(path.name, 0),
        })
    return {"documents": documents, "indexed_pages": len(pages)}


@app.get("/api/v1/knowledge/document-content")
def view_knowledge_document(request: Request, filename: str):
    _require_local_knowledge_admin(request)
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="无效的文件名。")
    docs_dir = Path(agent_instance.docs_dir).resolve()
    source = docs_dir / filename
    if (
        not source.is_file()
        or source.is_symlink()
        or source.suffix.lower() not in KNOWLEDGE_FILE_EXTENSIONS
        or source.name.lower().endswith((".pdf.txt", ".pdf.law.txt"))
    ):
        raise HTTPException(status_code=404, detail="未找到可查看的知识库资料。")

    media_types = {
        ".pdf": "application/pdf",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }
    return FileResponse(source, media_type=media_types[source.suffix.lower()])


@app.put("/api/v1/knowledge/document-content")
async def update_knowledge_document_content(
    request: Request,
    payload: Dict[str, Any] = Body(...),
):
    _require_local_knowledge_admin(request)
    source = _editable_knowledge_source(str(payload.get("filename", "")), {".txt", ".md"})
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=422, detail="资料内容必须是文本。")
    encoded_content = content.encode("utf-8")
    if len(encoded_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"资料内容超过限制（最大 {MAX_UPLOAD_BYTES} 字节）。")
    indexed_pages = await _replace_knowledge_source(source, encoded_content)
    return {"filename": source.name, "indexed_pages": indexed_pages, "message": "资料已保存并重新建立索引。"}


@app.put("/api/v1/knowledge/document-file")
async def replace_knowledge_document_file(
    request: Request,
    filename: str = Form(...),
    file: UploadFile = File(...),
):
    _require_local_knowledge_admin(request)
    source = _editable_knowledge_source(filename, {".pdf"})
    if Path(file.filename or "").suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="替换文件必须是 PDF。")
    temporary_path = source.parent / f".kb-upload-{uuid.uuid4().hex}.pdf"
    try:
        await save_upload_file(file, temporary_path, MAX_UPLOAD_BYTES)
        indexed_pages = await _replace_knowledge_source(source, temporary_path.read_bytes())
        return {"filename": source.name, "indexed_pages": indexed_pages, "message": "PDF 已替换并重新建立索引。"}
    finally:
        temporary_path.unlink(missing_ok=True)


@app.post("/api/v1/knowledge/documents")
async def upload_knowledge_document(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form(...),
):
    _require_local_knowledge_admin(request)
    if category not in KNOWLEDGE_CATEGORY_PREFIXES:
        raise HTTPException(status_code=422, detail="未知知识库分类。")
    raw_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(raw_name).suffix.lower()
    if extension not in KNOWLEDGE_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 PDF、TXT、MD 文件。")
    stem = re.sub(r"[^\w.-]+", "_", Path(raw_name).stem, flags=re.UNICODE).strip("._ ")
    if not stem:
        raise HTTPException(status_code=400, detail="文件名无有效内容。")

    docs_dir = Path(agent_instance.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if category == "law" and stem.lower().endswith(".law"):
        stem = stem[:-4]
    filename = (
        f"{KNOWLEDGE_CATEGORY_PREFIXES[category]}_{stem}.law.txt"
        if category == "law"
        else f"{KNOWLEDGE_CATEGORY_PREFIXES[category]}_{stem}{extension}"
    )
    destination = docs_dir / filename
    if destination.exists():
        raise HTTPException(status_code=409, detail="同名资料已存在，请先删除旧文件或更换文件名。")

    temporary_path = docs_dir / f".kb-upload-{uuid.uuid4().hex}{extension}"
    try:
        await save_upload_file(file, temporary_path, MAX_UPLOAD_BYTES)
        if category == "law":
            converted = await asyncio.to_thread(
                convert_law_source_to_txt,
                temporary_path,
                destination,
                raw_name,
            )
            if not converted:
                raise HTTPException(status_code=422, detail="法规文件无法提取有效文本，未加入知识库。")
            temporary_path.unlink(missing_ok=True)
        else:
            os.replace(temporary_path, destination)
        indexed_pages = await asyncio.to_thread(_reload_knowledge_index)
        return {
            "filename": filename,
            "indexed_pages": indexed_pages,
            "message": "资料已加入知识库并完成索引。",
        }
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if destination.exists():
            destination.unlink()
        destination.with_suffix(".articles.json").unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"资料索引失败：{exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


@app.delete("/api/v1/knowledge/documents")
def delete_knowledge_document(request: Request, filename: str):
    _require_local_knowledge_admin(request)
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="无效的文件名。")
    docs_dir = Path(agent_instance.docs_dir).resolve()
    source = docs_dir / filename
    if (
        not source.is_file()
        or source.is_symlink()
        or source.suffix.lower() not in KNOWLEDGE_FILE_EXTENSIONS
        or source.name.lower().endswith((".pdf.txt", ".pdf.law.txt"))
    ):
        raise HTTPException(status_code=404, detail="未找到可管理的知识库资料。")

    artifacts = {source.with_suffix(".articles.json")}
    if source.name.lower().endswith(".law.txt"):
        artifacts.update(law_text_related_files(source))
    elif source.suffix.lower() == ".pdf":
        for cache_path in {pdf_text_cache_path(source), source.with_suffix(".pdf.txt")}:
            artifacts.update([cache_path, cache_path.with_suffix(cache_path.suffix + ".meta.json")])
    source.unlink()
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)
    return {"indexed_pages": _reload_knowledge_index(), "message": "资料已删除并刷新索引。"}


@app.post("/api/v1/knowledge/reload")
def reload_knowledge_index(request: Request):
    _require_local_knowledge_admin(request)
    try:
        return {"indexed_pages": _reload_knowledge_index(), "message": "知识库索引已刷新。"}
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"知识库索引刷新失败：{exc}") from exc


@app.get("/api/v1/knowledge/rules")
def list_knowledge_rules(request: Request):
    _require_local_knowledge_admin(request)
    return {"rules": _knowledge_rule_book().list_all_rules()}


@app.put("/api/v1/knowledge/rules")
def save_knowledge_rule(request: Request, payload: Dict[str, Any] = Body(...)):
    _require_local_knowledge_admin(request)
    required = ("topic", "keywords", "risk_level", "standard_requirement")
    values = {name: str(payload.get(name, "")).strip() for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise HTTPException(status_code=422, detail=f"必填字段缺失：{', '.join(missing)}")
    saved = _knowledge_rule_book().upsert_rule(
        **values,
        forbidden_pattern=str(payload.get("forbidden_pattern", "")).strip(),
        recommended_clause=str(payload.get("recommended_clause", "")).strip(),
    )
    if not saved:
        raise HTTPException(status_code=500, detail="规则保存失败。")
    return {"message": "规则已保存。"}


@app.delete("/api/v1/knowledge/rules")
def delete_knowledge_rule(request: Request, topic: str):
    _require_local_knowledge_admin(request)
    if not topic.strip():
        raise HTTPException(status_code=422, detail="规则主题不能为空。")
    deleted = _knowledge_rule_book().delete_rule_by_topic(topic)
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到该规则。")
    return {"message": "规则已删除。"}


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
    return EventSourceResponse(
        review_orchestrator.stream(
            request.contract_text,
            limit_turns,
            task_label,
            request.review_side,
        )
    )


def main():
    print(f"[*] 正在启动 Legal Agent FastAPI Gateway 监听: http://{GATEWAY_HOST}:{GATEWAY_PORT}")
    uvicorn.run(
        app,
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
