#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: web/app_ui.py
职责:
1. 基于 Streamlit 构建全宽工业级法务合同审查专业工作台
2. 展示 ReAct 流式输出与工具调用结果
3. 支持多线程并发审查，各条款状态独立追踪
"""

import sys
import time
import os
import subprocess
import psutil
import html
import hashlib
import json
import re
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from markdown_it import MarkdownIt
import httpx
from core.prompts import RISK_LEVEL_REPORT_LEGEND
from core.schemas import ContractReviewReport
from services.report_parser import (
    clean_report_content,
    normalize_report_structure,
    parse_structured_report,
    render_report_article,
)
from services.review_execution import (
    execute_review_unit,
)
from web.api_client import check_gateway_health as fetch_gateway_health
from web.api_client import upload_contract_file as send_contract_file
from web.contract_client import rewrite_contract
from web.report_exporter import report_to_docx

API_BASE_URL = "http://127.0.0.1:9000"
_MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False})
GATEWAY_SCRIPT = ROOT_DIR / "gateway" / "api_server.py"
GATEWAY_LOG = ROOT_DIR / "data" / "gateway.log"
_gateway_process: Optional[subprocess.Popen] = None

# ==================== 1. 页面配置与样式注入 ====================
st.set_page_config(
    page_title="Legal Agent Lab | 合同智能审查工作台",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stHeader"] { position: static !important; }
    [data-testid="stSidebar"] { display: none !important; }
    @media (max-width: 768px) {
        [data-testid="stSidebar"] { display: block !important; }
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 12rem !important;
        max-width: 98% !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: inherit;
        margin-top: 0.1rem;
        margin-bottom: 0.2rem;
        line-height: 1.3;
    }
    .sub-header {
        font-size: 0.92rem;
        color: inherit;
        opacity: 0.72;
        margin-bottom: 0.8rem;
    }
    .circuit-break-card {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 14px;
        color: #92400E;
    }
    .thought-scroll-box {
        height: 240px;
        overflow-y: auto;
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Consolas', 'Courier New', Courier, monospace;
        font-size: 0.88rem;
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #334155;
        display: flex;
        flex-direction: column-reverse;
        margin-bottom: 12px;
    }
    .thought-content {
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.55;
    }
    .meta-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-family: monospace;
        background-color: #E2E8F0;
        color: #334155;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .report-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 2. Session 状态初始化 ====================
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "clauses" not in st.session_state:
    st.session_state.clauses = []
if "full_contract_text" not in st.session_state:
    st.session_state.full_contract_text = ""
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "structured_report" not in st.session_state:
    st.session_state.structured_report = None
if "evidence_records" not in st.session_state:
    st.session_state.evidence_records = {}
if "is_reviewing" not in st.session_state:
    st.session_state.is_reviewing = False
if "revised_contract" not in st.session_state:
    st.session_state.revised_contract = None
if "revised_contract_signature" not in st.session_state:
    st.session_state.revised_contract_signature = None
if "review_flow" not in st.session_state:
    st.session_state.review_flow = None


# ==================== 3. 辅助函数 ====================
def check_gateway_health() -> Dict[str, Any]:
    try:
        return fetch_gateway_health(API_BASE_URL)
    except Exception:
        return {"status": "unreachable"}


def reset_contract_revision_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("rewrite_clause_"):
            st.session_state.pop(key, None)
    st.session_state.revised_contract = None
    st.session_state.revised_contract_signature = None


def start_gateway_process() -> subprocess.Popen:
    """Start the local API gateway from the Streamlit process."""
    global _gateway_process
    if _gateway_process is not None and _gateway_process.poll() is None:
        return _gateway_process
    GATEWAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(GATEWAY_LOG, "a", encoding="utf-8")
    kwargs = {
        "cwd": str(ROOT_DIR),
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    _gateway_process = subprocess.Popen(
        [sys.executable, str(GATEWAY_SCRIPT)], **kwargs
    )
    # The child owns the duplicated file descriptor after Popen returns.
    log_handle.close()
    return _gateway_process


def restart_gateway_process() -> subprocess.Popen:
    """Restart this project's gateway, whether page-managed or terminal-started."""
    global _gateway_process
    response = httpx.get(f"{API_BASE_URL}/openapi.json", timeout=3.0)
    response.raise_for_status()
    if response.json().get("info", {}).get("title") != "Legal Agent Lab API Gateway":
        raise RuntimeError("9000 端口运行的服务不是本项目网关，已取消重启。")

    gateway_port = int(API_BASE_URL.rsplit(":", 1)[1])
    listener_pids = {
        connection.pid
        for connection in psutil.net_connections(kind="inet")
        if connection.status == psutil.CONN_LISTEN
        and connection.laddr
        and connection.laddr.port == gateway_port
        and connection.pid
        and connection.pid != os.getpid()
    }
    if _gateway_process is not None and _gateway_process.poll() is None:
        listener_pids.add(_gateway_process.pid)
    if not listener_pids:
        raise RuntimeError("无法定位网关进程；请检查当前用户是否有权限管理该进程。")

    processes = []
    for pid in listener_pids:
        try:
            process = psutil.Process(pid)
            process.terminate()
            processes.append(process)
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        process.kill()
    if alive:
        psutil.wait_procs(alive, timeout=3)
    _gateway_process = None

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        still_listening = any(
            connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and connection.laddr.port == gateway_port
            for connection in psutil.net_connections(kind="inet")
        )
        if not still_listening:
            break
        time.sleep(0.2)
    return start_gateway_process()


def upload_contract_file(uploaded_file, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        return send_contract_file(API_BASE_URL, uploaded_file, session_id)
    except Exception as e:
        st.error(f"上传文件网络请求失败: {e}")
    return None


def render_thought_box(text: str) -> str:
    escaped_text = html.escape(text) if text else "等待模型推演与法条检索..."
    return f"""
    <div class="thought-scroll-box">
        <div class="thought-content">{escaped_text}</div>
    </div>
    """


def clean_reasoning_text(text: str) -> str:
    """将模型文本中的 Markdown 标记移除，供纯文本过程框显示。"""
    visible_text = text or ""
    visible_text = re.sub(r"(?im)^\s*```(?:markdown|md)?\s*$", "", visible_text)
    visible_text = re.sub(r"(?im)^\s*```\s*$", "", visible_text)
    visible_text = re.sub(r"^\s{0,3}#{1,6}\s*", "", visible_text, flags=re.MULTILINE)
    visible_text = re.sub(r"^\s*[-*+]\s+", "• ", visible_text, flags=re.MULTILINE)
    visible_text = re.sub(r"(\*\*|__|`)", "", visible_text)
    visible_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", visible_text)
    return visible_text.strip()


def render_reasoning_text(text: str) -> str:
    """将模型文本显示为纯文本，避免 Markdown 语法直接出现在过程框。"""
    return render_thought_box(
        clean_reasoning_text(text) or "等待模型交互事件..."
    )


def report_text_for_display(text: str) -> str:
    """Hide the report legend from UI views while retaining it in stored reports."""
    content = clean_report_content(text)
    if content.startswith(RISK_LEVEL_REPORT_LEGEND):
        return content[len(RISK_LEVEL_REPORT_LEGEND):].lstrip()
    return content


def render_model_output_box(
    events: List[str], model_text: str, model_report: str
) -> str:
    sections = []
    if events:
        sections.append("审查进度：\n" + "\n".join(events))
    if model_text:
        sections.append("模型交互：\n" + clean_reasoning_text(model_text))
    if model_report:
        sections.append("报告生成中：\n" + model_report)
    return render_thought_box("\n\n".join(sections) or "等待审查进度...")


def render_saved_review_flow() -> None:
    flow = st.session_state.get("review_flow")
    if not flow:
        return

    st.markdown(f"##### 审查进度 · {flow.get('status', '审查中')}")
    if flow.get("mode") == "stream":
        st.markdown(f"**审查对象：{flow.get('title', '合同正文')}**")
        model_report = (
            "" if flow.get("status") == "审查完成"
            else flow.get("model_report", "")
        )
        rendered_output = render_model_output_box(
            flow.get("events", []),
            flow.get("thought", ""),
            model_report,
        )
        st.markdown(rendered_output, unsafe_allow_html=True)
        return

    for clause in flow.get("clauses", {}).values():
        st.markdown(f"**{clause.get('status', '审查中')}：{clause.get('title', '')}**")
        model_report = (
            "" if clause.get("status") == "审查完成"
            else clause.get("model_report", clause.get("report", ""))
        )
        rendered_output = render_model_output_box(
            clause.get("flow", []),
            clause.get("thought", ""),
            model_report,
        )
        st.markdown(rendered_output, unsafe_allow_html=True)


@st.dialog("条款内容")
def show_clause_content(title: str, content: str) -> None:
    """Display the complete clause text in a modal dialog."""
    st.markdown(f"#### {title}")
    st.caption(f"以下是单条审查提交给大模型的完整正文（{len(content)} 字符）。")
    st.text_area(
        "单条审查发送正文",
        value=content or "该条款暂无正文内容。",
        height=360,
        disabled=True,
        label_visibility="collapsed",
    )


def clause_review_text(clause: Dict[str, Any]) -> str:
    """Return the canonical clause text used both for display and model input."""
    raw_text = clause.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text.strip()
    title = str(clause.get("title", "")).strip()
    content = str(clause.get("content", "")).strip()
    if title and content and not content.startswith(title):
        return f"{title}\n{content}"
    return content or title


@st.dialog("依据原文")
def show_evidence_content(evidence: Dict[str, Any]) -> None:
    article = evidence.get("article_no") or "文档片段"
    title = evidence.get("title")
    st.markdown(f"#### {evidence.get('doc_name', '参考文档')} · {article}{f' {title}' if title else ''}")
    if evidence.get("source_location"):
        st.caption(f"知识库位置：{evidence['source_location']}")
    elif evidence.get("source_path"):
        st.caption(f"知识库位置：{evidence['source_path']}")
    if evidence.get("source_type") == "enterprise_rule":
        internal_grade = evidence.get("enterprise_risk_level")
        grade_label = f" · 企业内部风险等级：{internal_grade}" if internal_grade else ""
        st.caption(f"来源类型：企业自编规则（内部审查偏好，不是法律依据）{grade_label}")
    else:
        if not evidence.get("source_location"):
            st.caption(f"来源页码：{evidence.get('page_start', '?')}" + (f"–{evidence['page_end']}" if evidence.get("page_end") != evidence.get("page_start") else ""))
    st.text_area("命中条款原文（文本提取）", value=evidence.get("text", ""), height=420, disabled=True)


@st.dialog("完整审查报告", width="large")
def show_full_review_report(report: str) -> None:
    st.markdown(report)


def format_report_evidence_refs(
    report: str,
    evidence_records: Optional[Dict[str, Dict[str, Any]]] = None,
    inline_sources: bool = False,
) -> str:
    """Replace internal evidence IDs with readable citations and optional inline sources."""
    records = evidence_records or {}
    document_titles = {
        "minfadian": "中华人民共和国民法典",
        "minshishusongfa": "中华人民共和国民事诉讼法",
        "zhongcaifa": "中华人民共和国仲裁法",
    }

    def replace_reference(match: re.Match) -> str:
        evidence = records.get(match.group(1) or match.group(2) or match.group(3), {})
        raw_name = str(evidence.get("doc_name", "")).strip()
        if not raw_name:
            return ""
        stem = Path(raw_name).stem
        if stem.lower().endswith(".pdf"):
            stem = Path(stem).stem
        title = document_titles.get(stem.lower(), stem or "检索依据")
        citation = f"《{title}》"
        if evidence.get("title") and evidence.get("source_type") == "enterprise_rule":
            citation += f" · {evidence['title']}"
        if evidence.get("source_location"):
            citation += f"（{evidence['source_location']}）"
        elif evidence.get("source_path"):
            citation += f"（{evidence['source_path']}）"
        else:
            page_start = evidence.get("page_start")
            page_end = evidence.get("page_end", page_start)
            if page_start is not None:
                page_label = f"第 {page_start} 页"
                if page_end is not None and page_end != page_start:
                    page_label += f"–{page_end} 页"
                citation += f"（{page_label}）"
        if not inline_sources:
            return citation
        raw_source_text = str(evidence.get("text") or "未找到原文内容")
        is_markdown = Path(raw_name).suffix.lower() == ".md"
        source_text = (
            _MARKDOWN_RENDERER.render(raw_source_text)
            if is_markdown else html.escape(raw_source_text)
        )
        source_style = "white-space:normal;" if is_markdown else "white-space:pre-wrap;"
        return (
            '<details style="display:inline-block; vertical-align:baseline;">'
            '<summary style="display:inline; cursor:pointer; color:#126e76; text-decoration:underline;">'
            f"{html.escape(citation)}</summary>"
            f'<div style="{source_style} padding:0.5rem 0.7rem; margin:0.35rem 0; '
            'border-left:2px solid #126e76; background:#f3f8f8;">'
            f"{source_text}</div></details>"
        )

    reference_pattern = (
        r"\[\[EVIDENCE:(EV[A-Za-z0-9]+)\]\]|"
        r"\[\[RULE:(RULE\d+)\]\]|\[\[KB:(EV[A-Za-z0-9]+)\]\]"
    )
    return re.sub(reference_pattern, replace_reference, report or "")


# ==================== 4. 顶部控制栏 ====================
header_col, status_col = st.columns([3, 1])
with header_col:
    st.markdown('<div class="main-header">本地法务合同审查 Agent 工作台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">全宽自适应可视 · 层次化条款切片 · 思考流追踪 · 结构化审查报告</div>', unsafe_allow_html=True)

with status_col:
    health = check_gateway_health()

try:
    runtime_response = httpx.get(f"{API_BASE_URL}/api/v1/model-runtime", timeout=3.0)
    runtime_response.raise_for_status()
    runtime_status = runtime_response.json()
    models_response = httpx.get(f"{API_BASE_URL}/api/v1/models", timeout=3.0)
    models_response.raise_for_status()
    available_models = models_response.json().get("models", [])
except httpx.HTTPError:
    runtime_status, available_models = {"state": "unavailable"}, []

if health.get("status") == "healthy":
    if runtime_status.get("state") == "running":
        status_col.success(f"🟢 服务就绪 ({runtime_status.get('model') or health.get('model', 'qwen2.5')})")
    elif runtime_status.get("state") == "starting":
        status_col.info("🟡 模型服务启动中")
    else:
        status_col.warning("🟠 网关已启动，模型服务未运行")
else:
    status_col.error("🔴 网关未启动 (请运行 api_server.py)")

with st.expander("🛠️ 服务管理", expanded=health.get("status") != "healthy"):
    gateway_tab, model_tab = st.tabs(["网关服务", "推理模型"])
    with gateway_tab:
        if health.get("status") == "healthy":
            st.success(f"网关运行中：{API_BASE_URL}")
        elif _gateway_process is not None and _gateway_process.poll() is None:
            st.info("网关正在启动，请稍后刷新状态。")
        else:
            st.caption("启动本地 FastAPI 网关后，可在推理模型页启动和切换模型。")

        gateway_start_col, gateway_restart_col, gateway_refresh_col = st.columns([1, 1, 1])
        with gateway_start_col:
            if st.button(
                "启动网关",
                type="primary",
                use_container_width=True,
                disabled=health.get("status") == "healthy" or (
                    _gateway_process is not None and _gateway_process.poll() is None
                ),
            ):
                try:
                    start_gateway_process()
                    st.rerun()
                except OSError as exc:
                    st.error(f"网关启动失败：{exc}")
        with gateway_restart_col:
            if st.button(
                "重启网关",
                use_container_width=True,
                disabled=health.get("status") != "healthy",
            ):
                try:
                    restart_gateway_process()
                    st.toast("网关已重启，稍后刷新状态。")
                    st.rerun()
                except (httpx.HTTPError, psutil.Error, OSError, RuntimeError) as exc:
                    st.error(f"网关重启失败：{exc}")
        with gateway_refresh_col:
            if st.button("刷新网关状态", use_container_width=True):
                st.rerun()
        st.caption("网关日志：data/gateway.log")

    with model_tab:
        if available_models:
            model_keys = [item["key"] for item in available_models]
            selected_model = st.selectbox(
                "推理模型",
                model_keys,
                format_func=lambda key: next(
                    f"{item['name']} · {item['context']:,} tokens" for item in available_models if item["key"] == key
                ),
                index=model_keys.index(runtime_status["model_key"])
                if runtime_status.get("model_key") in model_keys else 0,
            )
            current_state = runtime_status.get("state", "unknown")
            if current_state == "running":
                st.success(f"模型服务运行中：{runtime_status.get('model')}")
            elif current_state == "starting":
                st.info("模型正在启动或下载权重；点击刷新查看状态。")
            else:
                st.caption("模型服务已停止。启动后即可进行合同审查。")

            start_col, stop_col, refresh_col = st.columns([1, 1, 1])
            with start_col:
                if st.button("启动 / 切换模型", type="primary", use_container_width=True):
                    try:
                        response = httpx.post(
                            f"{API_BASE_URL}/api/v1/model-runtime/start",
                            json={"model_key": selected_model},
                            timeout=10.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.session_state.model_runtime_notice = (
                            "模型服务正在启动；首次启动可能需要下载权重。" if result.get("state") == "starting"
                            else f"已切换到 {result.get('model', selected_model)}。"
                        )
                        st.rerun()
                    except httpx.HTTPError as exc:
                        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
                        st.error(f"模型启动失败：{detail}")
            with stop_col:
                if st.button("停止模型服务", use_container_width=True, disabled=current_state != "running"):
                    try:
                        response = httpx.post(f"{API_BASE_URL}/api/v1/model-runtime/stop", timeout=12.0)
                        response.raise_for_status()
                        st.session_state.model_runtime_notice = "模型服务已停止。"
                        st.rerun()
                    except httpx.HTTPError as exc:
                        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
                        st.error(f"停止失败：{detail}")
            with refresh_col:
                if st.button("刷新模型状态", use_container_width=True):
                    st.rerun()
            if notice := st.session_state.pop("model_runtime_notice", None):
                st.toast(notice)
            st.caption("启动参数：`python server/vllm_launcher.py --model <所选模型> --no-fp8-kv`；日志：data/vllm.log")
        else:
            st.warning("请先在网关服务页启动网关，待其运行后即可选择模型。")

with nullcontext():
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(
        [1.5, 1.1, 0.9, 1.0, 0.9], gap="medium"
    )

    with ctrl_col1:
        review_mode = st.radio(
            "审查策略：",
            ["分条款并发精审 (推荐·高效防截断)", "全篇审查"],
            horizontal=False,
            index=0,
        )

    with ctrl_col2:
        concurrency = st.slider(
            "⚡ 并发线程数",
            min_value=1,
            max_value=100,
            value=8,
            step=1,
            disabled=review_mode == "全篇审查",
        )

    with ctrl_col3:
        max_turns = st.slider("每轮最大探索步数", min_value=10, max_value=30, value=10, step=1)

    with ctrl_col4:
        review_side = st.selectbox(
            "审查立场",
            ["neutral", "buyer", "seller"],
            format_func=lambda side: {"neutral": "中立", "buyer": "甲方", "seller": "乙方"}[side],
        )

    with ctrl_col5:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if st.button("📋 载入甲乙方样例", use_container_width=True):
            sample_clauses = [
                {"index": 1, "type": "preamble", "title": "合同前言与标题", "content": "高端智能制造设备采购与长期技术维保协议"},
                {"index": 2, "type": "article", "title": "第一条 交付期限与违约金", "content": "乙方逾期交付的，每日应按合同总金额的 5% 向甲方支付惩罚性违约金。"},
                {"index": 3, "type": "article", "title": "第二条 争议管辖与独任仲裁", "content": "因本合同发生的一切争议，由甲方指定的独任仲裁员在其个人办公场所秘密裁决，裁决为终局。"}
            ]
            reset_contract_revision_state()
            st.session_state.clauses = sample_clauses
            st.session_state.full_contract_text = "\n\n".join([f"{c['title']}\n{c['content']}" for c in sample_clauses])
            st.session_state.final_report = ""
            st.session_state.structured_report = None
            st.session_state.evidence_records = {}
            st.session_state.revised_contract = None
            st.session_state.revised_contract_signature = None
            st.session_state.review_flow = None
            st.rerun()

# ==================== 5. 中部工作区 ====================
with nullcontext():
    st.markdown("#### 📄 待审查合同数据源")
    top_col1, top_col2 = st.columns([1, 2], gap="medium")
    with top_col1:
        uploaded_file = st.file_uploader(
            "上传合同文件 (Word / PDF / 图片 / TXT / Markdown)",
            type=["docx", "doc", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff", "txt", "md"],
        )
        upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
        upload_signature = (
            hashlib.sha256(
                uploaded_file.name.encode("utf-8") + b"\0" + upload_bytes
            ).hexdigest() if uploaded_file is not None else None
        )
        if uploaded_file is not None and st.session_state.get("last_uploaded_signature") != upload_signature:
            with st.spinner("正在执行多模态解析与条款切分..."):
                res = upload_contract_file(uploaded_file, st.session_state.session_id)
                if res and res.get("code") == 200:
                    data = res["data"]
                    reset_contract_revision_state()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.clauses = data["clauses"]
                    st.session_state.full_contract_text = "\n\n".join([f"{c['title']}\n{c['content']}" for c in data["clauses"]])
                    st.session_state.last_uploaded_signature = upload_signature
                    st.session_state.final_report = ""
                    st.session_state.structured_report = None
                    st.session_state.evidence_records = {}
                    st.session_state.revised_contract = None
                    st.session_state.revised_contract_signature = None
                    st.session_state.review_flow = None
                    st.success(f"解析成功，切分出 {len(data['clauses'])} 个条款单元！")
                    st.rerun()

    with top_col2:
        st.text_area("合同原文预览（只读）：", value=st.session_state.full_contract_text, height=130, disabled=True)

    selected_clause_idx = None
    if st.session_state.clauses:
        with st.expander(
            f"📑 条款明细与合同修订 (共 {len(st.session_state.clauses)} 个单元)",
            expanded=False,
        ):
            st.caption("点击条款标题查看全文；勾选需要根据审查意见修订的条款。导出的修订版是按解析文本重新排版的草稿，不保留原 Word/PDF 页面布局。")
            clause_cols = st.columns(2, gap="medium")
            for idx, c in enumerate(st.session_state.clauses):
                col_target = clause_cols[idx % 2]
                with col_target:
                    c_col1, c_col2 = st.columns([4, 1])
                    with c_col1:
                        clause_index = c.get("index")
                        clause_title = c.get("title", f"条款 {clause_index}")
                        if st.button(
                            f"#{clause_index} {clause_title}",
                            key=f"btn_clause_content_{clause_index}",
                            use_container_width=True,
                        ):
                            show_clause_content(
                                clause_title,
                                clause_review_text(c),
                            )
                    with c_col2:
                        if st.button("单独审查", key=f"btn_single_{c.get('index')}"):
                            selected_clause_idx = c.get("index")
                    st.caption(clause_review_text(c)[:120] + ("..." if len(clause_review_text(c)) > 120 else ""))
                    st.checkbox(
                        "纳入合同修订",
                        key=f"rewrite_clause_{c.get('index')}",
                        disabled=st.session_state.is_reviewing,
                    )
                    st.divider()

            selected_revision_indices = [
                int(c.get("index"))
                for c in st.session_state.clauses
                if st.session_state.get(f"rewrite_clause_{c.get('index')}", False)
            ]
            revision_signature = (
                tuple(selected_revision_indices),
                st.session_state.final_report,
            )
            if not st.session_state.final_report:
                st.info("完成合同审查后，可在此选择条款并生成修订合同。")
            elif not selected_revision_indices:
                st.warning("请选择至少一个“纳入合同修订”的条款。")
            else:
                if st.button(
                    "根据选中条款和审查意见生成修订合同",
                    key="generate_revised_contract",
                    type="primary",
                    disabled=st.session_state.is_reviewing,
                ):
                    st.session_state.revised_contract = None
                    st.session_state.revised_contract_signature = None
                    with st.spinner("正在按选中条款生成修订合同..."):
                        try:
                            review_article = (
                                render_report_article(
                                    ContractReviewReport.model_validate(
                                        st.session_state.structured_report
                                    )
                                )
                                if st.session_state.structured_report
                                else st.session_state.final_report
                            )
                            review_article = format_report_evidence_refs(
                                review_article,
                                st.session_state.evidence_records,
                            )
                            st.session_state.revised_contract = rewrite_contract(
                                API_BASE_URL,
                                st.session_state.clauses,
                                review_article,
                                selected_revision_indices,
                            )
                            st.session_state.revised_contract_signature = revision_signature
                            st.success("修订合同生成完成，未选中的条款保持原文。")
                        except Exception as exc:
                            st.error(f"生成修订合同失败: {exc}")

            if (
                st.session_state.revised_contract
                and st.session_state.revised_contract_signature == revision_signature
            ):
                st.markdown("#### 📝 修订文本草稿预览")
                st.text_area(
                    "修订文本草稿正文",
                    value=st.session_state.revised_contract["contract_text"],
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                )
                revised_export_col1, revised_export_col2 = st.columns(2)
                revised_filename = f"revised_contract_{int(time.time())}"
                with revised_export_col1:
                    st.download_button(
                        "📥 导出修订文本草稿 Markdown",
                        data=st.session_state.revised_contract["contract_text"],
                        file_name=f"{revised_filename}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with revised_export_col2:
                    try:
                        revised_docx = report_to_docx(
                            st.session_state.revised_contract["contract_text"]
                        )
                        st.download_button(
                            "📄 导出修订文本草稿 Word",
                            data=revised_docx,
                            file_name=f"{revised_filename}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))

    start_full_review = st.button("🚀 开始执行合同智能合规审查", type="primary", use_container_width=True, disabled=st.session_state.is_reviewing or not bool(st.session_state.full_contract_text))

st.markdown("<hr style='margin: 1.4rem 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)


# ==================== 6. 单通道 SSE 审查逻辑 ====================
def execute_stream_review(
    text_to_review: str,
    target_name: str = "合同正文",
    review_run_id: Optional[str] = None,
    review_side: str = "neutral",
) -> Optional[str]:
    st.session_state.is_reviewing = True
    st.session_state.final_report = ""
    st.session_state.structured_report = None
    saved_flow = {
        "mode": "stream",
        "title": target_name,
        "status": "审查中",
        "thought": "",
        "report": "",
        "model_report": "",
        "events": [],
    }
    st.session_state.review_flow = saved_flow

    flow_entries = []
    accumulated_tokens = ""
    accumulated_report = ""

    with nullcontext():
        status_label = st.empty()
        status_label.markdown(f"**正在初始化 ReAct 推理链路 ({target_name})...**")

    st.caption(f"审查进度 [{target_name}]")
    thought_container = st.empty()

    def refresh_thought_box() -> None:
        thought_container.markdown(
            render_model_output_box(
                flow_entries,
                accumulated_tokens,
                accumulated_report,
            ),
            unsafe_allow_html=True,
        )

    def append_flow(entry: str) -> None:
        flow_entries.append(entry)
        saved_flow["events"] = flow_entries.copy()
        refresh_thought_box()

    refresh_thought_box()
    review_run_id = review_run_id or uuid.uuid4().hex

    def handle_event(event: str, data: Dict[str, Any]) -> None:
        nonlocal accumulated_tokens, accumulated_report
        if event == "start":
            status_label.markdown(f"**[{data.get('task_id', target_name)}] 正在制定审查策略...**")
        elif event == "token":
            accumulated_tokens += data.get("token", "")
            saved_flow["thought"] = accumulated_tokens
            refresh_thought_box()
        elif event == "report_token":
            accumulated_report += data.get("token", "")
            saved_flow["model_report"] = accumulated_report
            refresh_thought_box()
        elif event == "tool_start":
            query = data.get("query")
            append_flow(
                f"调度工具：{data.get('tool')}"
                + (f"；检索词：{query}" if query else "")
            )
        elif event == "tool_result":
            if data.get("success"):
                sources = data.get("evidence_sources", [])
                source_text = "；命中来源：" + "；".join(
                    f"{item.get('source_location', '来源位置未提供')} [{item.get('evidence_id')}]"
                    for item in sources
                ) if sources else "；未命中知识库来源"
                append_flow(
                    f"工具结果：已注入模型工作记忆（带出 {data.get('injected_chars', 0)} 字符）"
                    f"{source_text}"
                )
            else:
                append_flow(
                    f"工具调用失败，错误已交回模型修正：{data.get('error', '')}"
                )
        elif event == "pipeline_stage":
            if data.get("stage") == "tool_recording":
                return
            message = data.get("message") or ""
            if data.get("stage") == "review_complete":
                message = "报告复核完成"
            if message:
                append_flow(message)
        elif event == "guardrail":
            for finding in data.get("findings", []):
                append_flow(
                    f"程序规则提示（需核实）：{finding.get('message')} "
                    f"证据片段：{finding.get('evidence')}"
                )
        elif event == "model_start":
            message = f"正在分析第 {data.get('turn')} 轮..."
            status_label.markdown(f"**{message}**")
        elif event == "error":
            error_text = f"审查失败：{data.get('error', '网关返回错误')}"
            status_label.markdown(f"**{error_text}**")
            append_flow(error_text)
        elif event == "final_report":
            status_label.markdown(
                "**正在校验最终审查报告...**"
                if data.get("is_complete")
                else "**审查未完整完成，结果未保存**"
            )

    result = execute_review_unit(
        API_BASE_URL,
        text_to_review,
        max_turns,
        review_run_id,
        review_side,
        on_event=handle_event,
    )
    st.session_state.evidence_records.update(result["evidence_records"])
    accumulated_tokens = result["model_text"]
    accumulated_report = result["model_report"] if not result["success"] else ""
    saved_flow["thought"] = accumulated_tokens
    saved_flow["model_report"] = accumulated_report
    refresh_thought_box()

    if not result["success"]:
        saved_flow["status"] = "审查失败"
        status_label.markdown(f"**审查失败：{result['error']}**")
        if not any(result["error"] in entry for entry in flow_entries):
            append_flow(f"审查失败：{result['error']}")
        st.session_state.is_reviewing = False
        return None

    formatted_report = result["report"]
    saved_flow["report"] = formatted_report
    parsed_report = parse_structured_report(formatted_report)
    st.session_state.structured_report = (
        parsed_report.model_dump(mode="json") if parsed_report else None
    )
    saved_flow["status"] = "审查完成"
    status_label.markdown("**审查完成！**")
    append_flow("审查完成")
    st.session_state.is_reviewing = False
    return formatted_report


# ==================== 7. 多线程并发调度器 ====================
def _worker_clause_review(
    clause: Dict[str, Any], turns: int, review_run_id: str, review_side: str,
    event_queue: Queue,
) -> Dict[str, Any]:
    clause_text = clause_review_text(clause)
    logs = []
    clause_index = clause.get("index", 0)

    def publish(kind: str, message: str) -> None:
        event_queue.put((clause_index, kind, message))

    def handle_event(event: str, data: Dict[str, Any]) -> None:
        if event == "start":
            publish("status", "正在制定审查策略")
        elif event == "token":
            publish("thought", data.get("token", ""))
        elif event == "report_token":
            publish("report", data.get("token", ""))
        elif event == "tool_start":
            message = (
                f"检索: {data.get('query')}" if data.get("query")
                else f"调用工具: {data.get('tool')}"
            )
            logs.append(message)
            publish("log", message)
        elif event == "tool_result":
            sources = data.get("evidence_sources", [])
            source_text = "；命中来源：" + "；".join(
                f"{item.get('source_location', '来源位置未提供')} [{item.get('evidence_id')}]"
                for item in sources
            ) if sources else "；未命中知识库来源"
            message = (
                f"工具完成: {data.get('tool')}；带出 {data.get('injected_chars', 0)} 字符"
                f"{source_text}"
                if data.get("success")
                else f"工具失败并已回传模型: {data.get('error', '')}"
            )
            logs.append(message)
            publish("log", message)
        elif event == "pipeline_stage":
            if data.get("stage") == "tool_recording":
                return
            message = data.get("message") or ""
            if data.get("stage") == "review_complete":
                message = "报告复核完成"
            if message:
                logs.append(message)
                publish("log", message)
        elif event == "guardrail":
            for finding in data.get("findings", []):
                message = f"规则提示: {finding.get('message')}"
                logs.append(message)
                publish("log", message)
        elif event == "model_start":
            publish("status", f"正在分析第 {data.get('turn')} 轮")
        elif event == "error":
            message = data.get("error", "网关返回错误")
            logs.append(message)
            publish("error", message)

    result = execute_review_unit(
        API_BASE_URL,
        clause_text,
        turns,
        review_run_id,
        review_side,
        on_event=handle_event,
    )
    if not result["success"] and result["error"] not in logs:
        logs.append(result["error"])
        publish("error", result["error"])
    return {
        **result,
        "index": clause_index,
        "title": clause.get("title", ""),
        "logs": logs,
    }


def execute_concurrent_clause_review(
    clauses_to_review: List[Dict[str, Any]], max_workers: int,
    review_run_id: Optional[str] = None,
    review_side: str = "neutral",
) -> str:
    st.session_state.is_reviewing = True
    st.session_state.structured_report = None
    st.session_state.evidence_records = {}
    review_run_id = review_run_id or uuid.uuid4().hex
    total_count = len(clauses_to_review)

    st.markdown(f"##### ⚡ 正在启用 {max_workers} 路线程并发审查 (共 {total_count} 个条款)...")
    progress_bar = st.progress(0, text=f"准备调度并发任务 (0/{total_count})...")

    status_placeholders = {}
    thought_placeholders = {}
    live_state = {}
    display_cols_count = min(4, max(2, min(total_count, max_workers)))
    st_cols = st.columns(display_cols_count, gap="small")
    for i, c in enumerate(clauses_to_review):
        col_target = st_cols[i % display_cols_count]
        with col_target:
            clause_index = c["index"]
            status_placeholders[clause_index] = st.empty()
            status_placeholders[clause_index].markdown(f"**排队中: {c['title']}**")
            thought_placeholders[clause_index] = st.empty()
            thought_placeholders[clause_index].markdown(
                render_reasoning_text(""), unsafe_allow_html=True
            )
            live_state[clause_index] = {
                "title": c.get("title", ""),
                "status": "排队中",
                "thought": "",
                "model_report": "",
                "report": "",
                "final_report": "",
                "flow": [],
            }

    results = []
    completed_count = 0
    event_queue = Queue()
    st.session_state.review_flow = {
        "mode": "parallel",
        "status": "并发审查中",
        "clauses": live_state,
    }

    def render_queued_events() -> None:
        changed_thoughts = set()
        changed_reports = set()
        changed_flows = set()
        while True:
            try:
                clause_index, kind, message = event_queue.get_nowait()
            except Empty:
                break
            state = live_state.get(clause_index)
            if state is None:
                continue
            if kind == "thought":
                state["thought"] += message
                changed_thoughts.add(clause_index)
            elif kind == "report":
                state["model_report"] += message
                changed_reports.add(clause_index)
            elif kind in {"log", "error"}:
                state["flow"].append(message)
                changed_flows.add(clause_index)
                if kind == "error":
                    state["status"] = "审查失败"
                    status_placeholders[clause_index].markdown(
                        f"**审查失败: {message}**"
                    )
            elif kind == "status":
                state["status"] = message
                status_placeholders[clause_index].markdown(f"**{message}**")
        for clause_index in changed_thoughts:
            thought_placeholders[clause_index].markdown(
                render_reasoning_text(live_state[clause_index]["thought"]),
                unsafe_allow_html=True,
            )
        for clause_index in changed_thoughts | changed_reports | changed_flows:
            state = live_state[clause_index]
            model_report = (
                "" if state["status"] == "审查完成"
                else state["model_report"]
            )
            thought_placeholders[clause_index].markdown(
                render_model_output_box(
                    state["flow"], state["thought"], model_report
                ),
                unsafe_allow_html=True,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _worker_clause_review,
                c,
                max_turns,
                review_run_id,
                review_side,
                event_queue,
            ): c["index"]
            for c in clauses_to_review
        }
        pending = set(future_map)
        while pending:
            render_queued_events()
            completed, pending = wait(
                pending,
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                c_idx = future_map[future]
                res = future.result()
                results.append(res)
                completed_count += 1

                st.session_state.evidence_records.update(res.get("evidence_records", {}))
                live_state[c_idx]["thought"] = res.get("model_text", "")
                live_state[c_idx]["model_report"] = res.get("model_report", "")
                live_state[c_idx]["report"] = res.get("report", "")
                live_state[c_idx]["final_report"] = res.get("report", "")
                live_state[c_idx]["status"] = (
                    "审查完成" if res["success"] else "审查失败"
                )
                status_placeholders[c_idx].markdown(
                    f"**{'审查完成' if res['success'] else '审查失败'}: {res['title']}**"
                )
                thought_placeholders[c_idx].markdown(
                    render_model_output_box(
                        live_state[c_idx]["flow"],
                        live_state[c_idx]["thought"],
                        "",
                    ),
                    unsafe_allow_html=True,
                )
                progress_bar.progress(
                    completed_count / total_count,
                    text=f"并发审查进度 ({completed_count}/{total_count})...",
                )

        render_queued_events()

    progress_bar.empty()
    success_count = sum(result["success"] for result in results)
    if results and success_count == len(results):
        overall_status = "并发审查完成"
    elif success_count:
        overall_status = "并发审查部分完成"
    else:
        overall_status = "并发审查失败"
    st.session_state.review_flow["status"] = overall_status
    results.sort(key=lambda x: x["index"])
    aggregated_reviews = []
    for result in results:
        if not result["success"]:
            continue
        parsed_report = parse_structured_report(result["report"])
        if parsed_report:
            aggregated_reviews.extend(
                item.model_dump(mode="json") for item in parsed_report.reviews
            )

    st.session_state.is_reviewing = False
    return json.dumps({"reviews": aggregated_reviews}, ensure_ascii=False)


# ==================== 8. 下部：审查轨迹与展示 ====================
with nullcontext():
    review_flow = st.session_state.get("review_flow")
    if review_flow:
        flow_status = review_flow.get("status", "审查中")
        with st.expander(
            f"🤖 模型交互记录 · {flow_status}",
            expanded=st.session_state.is_reviewing,
        ):
            render_saved_review_flow()

    if selected_clause_idx is not None:
        target_clause = next((c for c in st.session_state.clauses if c.get("index") == selected_clause_idx), None)
        if target_clause:
            st.session_state.evidence_records = {}
            target_text = clause_review_text(target_clause)
            report = execute_stream_review(
                target_text,
                target_name=target_clause["title"],
                review_side=review_side,
            )
            if report:
                st.session_state.final_report = report
                st.rerun()

    elif start_full_review:
        st.session_state.evidence_records = {}
        review_run_id = uuid.uuid4().hex
        if not st.session_state.full_contract_text.strip():
            st.warning("请先上传合同文件或载入样例数据！")
        elif review_mode == "全篇审查":
            st.session_state.final_report = ""
            report = execute_stream_review(
                st.session_state.full_contract_text,
                target_name="全篇合同",
                review_run_id=review_run_id,
                review_side=review_side,
            )
            if report:
                st.session_state.final_report = report
                st.rerun()
        else:
            article_clauses = [
                c for c in st.session_state.clauses if c.get("type") == "article"
            ] or st.session_state.clauses
            if not article_clauses:
                article_clauses = [{
                    "index": 1,
                    "type": "article",
                    "title": "全篇合同",
                    "content": st.session_state.full_contract_text,
                }]

            st.session_state.final_report = ""
            st.session_state.final_report = execute_concurrent_clause_review(
                article_clauses,
                max_workers=concurrency,
                review_run_id=review_run_id,
                review_side=review_side,
            )
            st.rerun()

    # 渲染 Markdown 报告
    if st.session_state.final_report:
        evidence_records = st.session_state.evidence_records
        parsed_report = parse_structured_report(st.session_state.final_report)
        st.session_state.structured_report = (
            parsed_report.model_dump(mode="json") if parsed_report else None
        )
        structured_reviews = (st.session_state.structured_report or {}).get("reviews", [])
        report_model = (
            ContractReviewReport.model_validate(st.session_state.structured_report)
            if st.session_state.structured_report is not None
            else None
        )
        export_source = (
            render_report_article(report_model)
            if report_model
            else report_text_for_display(st.session_state.final_report)
        )
        readable_report = format_report_evidence_refs(
            export_source,
            evidence_records,
        )
        export_filename = f"contract_review_{int(time.time())}"

        st.markdown("#### 🔍 风险审查报告")

        with st.container(border=True):
            summary_col, markdown_col, word_col, full_report_col = st.columns(
                [2, 1, 1, 1]
            )
            with summary_col:
                st.markdown("#### 审查结果")
                summary = (
                    f"共 {len(structured_reviews)} 条条款审查意见"
                    if report_model
                    else "报告暂无法解析为结构化条目"
                )
                st.caption(summary)
            with markdown_col:
                st.download_button(
                    label="📥 导出 Markdown",
                    data=readable_report,
                    file_name=f"{export_filename}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with word_col:
                try:
                    docx_report = report_to_docx(readable_report)
                    st.download_button(
                        label="📄 导出 Word",
                        data=docx_report,
                        file_name=f"{export_filename}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
            with full_report_col:
                if st.button("查看完整报告", use_container_width=True):
                    show_full_review_report(readable_report)

        if evidence_records:
            source_labels = {
                "law": "法规",
                "enterprise_document": "企业资料",
                "enterprise_rule": "企业规则",
                "general_document": "通用资料",
            }
            source_counts = {
                label: sum(item.get("source_type") == source_type for item in evidence_records.values())
                for source_type, label in source_labels.items()
            }
            source_summary = "，".join(
                f"{label} {count} 条" for label, count in source_counts.items() if count
            )
            with st.expander(
                f"📚 本次检索依据（共 {len(evidence_records)} 条：{source_summary}）",
                expanded=False,
            ):
                for evidence_id, evidence in evidence_records.items():
                    citation = f"《{evidence.get('doc_name', '参考文档')}》"
                    if evidence.get("article_no"):
                        citation += f" {evidence['article_no']}"
                    if evidence.get("title"):
                        citation += f" {evidence['title']}"
                    source_label = source_labels.get(evidence.get("source_type"), "资料")
                    citation = f"[{source_label}] {citation}"
                    if evidence.get("source_location"):
                        citation += f" · {evidence['source_location']}"
                    elif evidence.get("page_start") is not None:
                        citation += f" · 第 {evidence['page_start']} 页"
                    if evidence.get("enterprise_risk_level"):
                        citation += f" · 内部等级 {evidence['enterprise_risk_level']}"
                    if st.button(f"查看原文：{citation}", key=f"all_evidence_{evidence_id}"):
                        show_evidence_content(evidence)

        if report_model:
            st.caption(
                "按法律效力、商业后果、救济成本及所选审查立场综合分级；中风险及以上为审查红线；商务提示需要衡量履约能力。低风险区企业内部风险等级为高时需要重点关注！"
            )
            risk_levels = [
                ("High", "高风险"),
                ("Medium", "中风险"),
                ("Low", "低风险"),
                ("Notice", "履约/商务提示"),
            ]
            counts = {
                level: sum(
                    1 for item in structured_reviews
                    if str(item.get("risk_level", "")).lower() == level.lower()
                )
                for level, _ in risk_levels
            }
            risk_tabs = st.tabs([
                f"{label}（{counts[level]}）" for level, label in risk_levels
            ])
            for tab, (level, label) in zip(risk_tabs, risk_levels):
                with tab:
                    items = [
                        item for item in structured_reviews
                        if str(item.get("risk_level", "")).lower() == level.lower()
                    ]
                    if not items:
                        st.info(f"当前报告没有{label}条款。")
                    for index, item in enumerate(items, 1):
                        clause_title = item.get("clause_topic", "未命名条款")
                        with st.expander(
                            f"{index}. {clause_title} · {label}",
                            expanded=True,
                        ):
                            if item.get("risk_type"):
                                st.markdown(f"**风险类型：** {item['risk_type']}")
                            st.markdown(f"**风险等级：** {label}")
                            if item.get("enterprise_risk_level"):
                                st.markdown(f"**企业内部风险等级：** {item['enterprise_risk_level']}")
                            if item.get("affected_party"):
                                st.markdown(f"**受影响方：** {item['affected_party']}")
                            if item.get("confidence"):
                                st.markdown(f"**结论置信度：** {item['confidence']}")
                            for field, label_text in (
                                ("legal_effect", "法律效力"),
                                ("commercial_impact", "商业后果"),
                                ("remedy_cost", "救济成本"),
                            ):
                                if item.get(field):
                                    st.markdown(f"**{label_text}：** {item[field]}")
                            legal_basis = item.get("legal_basis", "")
                            evidence_ids = list(dict.fromkeys(re.findall(r"\[\[EVIDENCE:(EV[A-Za-z0-9]+)\]\]", legal_basis)))
                            legal_basis_for_display = legal_basis
                            if not evidence_ids and evidence_records:
                                article_refs = set(re.findall(r"第\s*[零〇一二三四五六七八九十百千万两0-9]+\s*条", legal_basis))
                                for article_ref in article_refs:
                                    matching_refs = []
                                    for evidence_id, evidence in evidence_records.items():
                                        if evidence.get("article_no") != article_ref:
                                            continue
                                        if re.fullmatch(r"EV[A-Za-z0-9]+", evidence_id):
                                            matching_refs.append(f"[[EVIDENCE:{evidence_id}]]")
                                        elif re.fullmatch(r"RULE\d+", evidence_id):
                                            matching_refs.append(f"[[RULE:{evidence_id}]]")
                                    if matching_refs:
                                        legal_basis_for_display = legal_basis_for_display.replace(
                                            article_ref, "、".join(matching_refs), 1
                                        )
                            visible_basis = format_report_evidence_refs(
                                legal_basis_for_display,
                                evidence_records,
                                inline_sources=True,
                            ).strip()
                            visible_basis = re.sub(r"^[\s、，,；;]+|[\s、，,；;]+$", "", visible_basis)
                            if visible_basis:
                                st.markdown(f"**法律/合规依据：** {visible_basis}", unsafe_allow_html=True)
                            elif evidence_ids:
                                st.markdown("**法律/合规依据：** 关联以下检索原文（请核对原文是否支持本项分析）")
                            else:
                                st.markdown("**法律/合规依据：** 未检索到直接依据")
                            enterprise_basis = format_report_evidence_refs(
                                item.get("enterprise_basis", ""),
                                evidence_records,
                                inline_sources=True,
                            ).strip()
                            if enterprise_basis:
                                st.markdown(f"**企业知识库依据：** {enterprise_basis}", unsafe_allow_html=True)
                            for evidence_id in evidence_ids:
                                if evidence_id not in evidence_records:
                                    st.warning(f"报告引用的证据 {evidence_id} 未在本次检索记录中找到。")
                            st.markdown(f"**风险分析：** {item.get('issue', '')}")
                            st.markdown(f"**修改建议：** {item.get('suggested_revision', '')}")
        else:
            st.warning("报告未能解析成风险条目，以下显示完整原始报告。")
            st.code(
                report_text_for_display(st.session_state.final_report),
                language="markdown",
                wrap_lines=True,
            )
    elif not st.session_state.is_reviewing:
        st.info("💡 操作指引：确认待审合同后，点击【开始执行合同智能合规审查】或展开条款明细点击【单独审查】。")
