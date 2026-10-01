#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: web/app_ui.py
职责:
1. 基于 Streamlit 构建全宽工业级法务合同审查专业工作台
2. 捕获后端 circuit_break 事件，在前端可视化展示具体的熔断原因与知识库增补推荐
3. 支持 1-100 路多线程并发审查 (Map-Reduce 架构)，各条款状态独立追踪
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
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import httpx
from services.report_parser import clean_report_content, parse_structured_report
from web.api_client import check_gateway_health as fetch_gateway_health
from web.api_client import upload_contract_file as send_contract_file
from web.contract_client import rewrite_contract
from web.report_exporter import report_to_docx
from web.sse_client import stream_contract_review

API_BASE_URL = "http://127.0.0.1:9000"
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
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 98% !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 0.1rem;
        margin-bottom: 0.2rem;
        line-height: 1.3;
    }
    .sub-header {
        font-size: 0.92rem;
        color: #64748B;
        margin-bottom: 0.8rem;
    }
    .top-control-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px 8px 18px;
        margin-bottom: 1.2rem;
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
if "consistency_checks" not in st.session_state:
    st.session_state.consistency_checks = []
if "is_reviewing" not in st.session_state:
    st.session_state.is_reviewing = False
if "truncation_warning" not in st.session_state:
    st.session_state.truncation_warning = None
if "latest_meta" not in st.session_state:
    st.session_state.latest_meta = None
if "circuit_breaks" not in st.session_state:
    st.session_state.circuit_breaks = []  # 存储各任务熔断详情
if "revised_contract" not in st.session_state:
    st.session_state.revised_contract = None
if "revised_contract_signature" not in st.session_state:
    st.session_state.revised_contract_signature = None


# ==================== 3. 辅助函数 ====================
def check_gateway_health() -> Dict[str, Any]:
    try:
        return fetch_gateway_health(API_BASE_URL)
    except Exception:
        return {"status": "unreachable"}


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


def render_reasoning_text(text: str) -> str:
    """将模型推理流转为纯文本，避免 Markdown 语法直接出现在思考区。"""
    visible_text = text or ""
    final_markers = ("Final:", "【最终结论】", "最终审查意见", "综合审查报告")
    marker_positions = [visible_text.find(marker) for marker in final_markers if visible_text.find(marker) >= 0]
    if marker_positions:
        visible_text = visible_text[:min(marker_positions)]

    visible_text = re.sub(r"(?im)^\s*```(?:markdown|md)?\s*$", "", visible_text)
    visible_text = re.sub(r"(?im)^\s*```\s*$", "", visible_text)
    visible_text = re.sub(r"^\s{0,3}#{1,6}\s*", "", visible_text, flags=re.MULTILINE)
    visible_text = re.sub(r"^\s*[-*+]\s+", "• ", visible_text, flags=re.MULTILINE)
    visible_text = re.sub(r"(\*\*|__|`)", "", visible_text)
    visible_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", visible_text)
    visible_text = visible_text.strip()
    return render_thought_box(visible_text or "模型正在整理最终审查报告...")


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
    st.caption(f"来源页码：{evidence.get('page_start', '?')}" + (f"–{evidence['page_end']}" if evidence.get("page_end") != evidence.get("page_start") else ""))
    st.text_area("命中条款原文（文本提取）", value=evidence.get("text", ""), height=420, disabled=True)


def collect_evidence(observation: str, destination: Optional[Dict[str, Any]] = None) -> None:
    """Read evidence records from the JSON returned by a knowledge search tool."""
    target = destination if destination is not None else st.session_state.evidence_records
    try:
        payload = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return
    for evidence in payload.get("evidence", []):
        if evidence.get("id"):
            target[evidence["id"]] = evidence


def format_report_evidence_refs(
    report: str, evidence_records: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """Replace internal evidence IDs with readable source titles and page numbers."""
    records = evidence_records or {}
    document_titles = {
        "minfadian": "中华人民共和国民法典",
        "minshishusongfa": "中华人民共和国民事诉讼法",
        "zhongcaifa": "中华人民共和国仲裁法",
    }

    def replace_reference(match: re.Match) -> str:
        evidence = records.get(match.group(1), {})
        raw_name = str(evidence.get("doc_name", "")).strip()
        if not raw_name:
            return ""
        stem = Path(raw_name).stem
        if stem.lower().endswith(".pdf"):
            stem = Path(stem).stem
        title = document_titles.get(stem.lower(), stem or "检索依据")
        page_start = evidence.get("page_start")
        page_end = evidence.get("page_end", page_start)
        if page_start is None:
            return f"《{title}》"
        page_label = f"第 {page_start} 页"
        if page_end is not None and page_end != page_start:
            page_label += f"–{page_end} 页"
        return f"《{title}》（{page_label}）"

    return re.sub(r"\[\[EVIDENCE:(EV[A-Za-z0-9]+)\]\]", replace_reference, report or "")


# ==================== 4. 顶部控制栏 ====================
header_col, status_col = st.columns([3, 1])
with header_col:
    st.markdown('<div class="main-header">⚖️ 本地法务合同审查 Agent 工作台</div>', unsafe_allow_html=True)
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

with st.container():
    st.markdown('<div class="top-control-panel">', unsafe_allow_html=True)
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
        max_turns = st.slider("每轮最大探索步数", min_value=3, max_value=30, value=10, step=1)

    with ctrl_col4:
        review_side = st.selectbox(
            "审查立场",
            ["neutral", "buyer", "seller"],
            format_func=lambda side: {"neutral": "中立", "buyer": "买方", "seller": "卖方"}[side],
        )

    with ctrl_col5:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if st.button("📋 载入买卖样例", use_container_width=True):
            sample_clauses = [
                {"index": 1, "type": "preamble", "title": "合同前言与标题", "content": "高端智能制造设备采购与长期技术维保协议"},
                {"index": 2, "type": "article", "title": "第一条 交付期限与违约金", "content": "乙方逾期交付的，每日应按合同总金额的 5% 向甲方支付惩罚性违约金。"},
                {"index": 3, "type": "article", "title": "第二条 争议管辖与独任仲裁", "content": "因本合同发生的一切争议，由甲方指定的独任仲裁员在其个人办公场所秘密裁决，裁决为终局。"}
            ]
            st.session_state.clauses = sample_clauses
            st.session_state.full_contract_text = "\n\n".join([f"{c['title']}\n{c['content']}" for c in sample_clauses])
            st.session_state.final_report = ""
            st.session_state.structured_report = None
            st.session_state.evidence_records = {}
            st.session_state.revised_contract = None
            st.session_state.revised_contract_signature = None
            st.session_state.circuit_breaks = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ==================== 5. 中部工作区 ====================
with st.container():
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
                    st.session_state.session_id = data["session_id"]
                    st.session_state.clauses = data["clauses"]
                    st.session_state.full_contract_text = "\n\n".join([f"{c['title']}\n{c['content']}" for c in data["clauses"]])
                    st.session_state.last_uploaded_signature = upload_signature
                    st.session_state.final_report = ""
                    st.session_state.structured_report = None
                    st.session_state.evidence_records = {}
                    st.session_state.revised_contract = None
                    st.session_state.revised_contract_signature = None
                    st.session_state.circuit_breaks = []
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
            if not st.session_state.final_report:
                st.info("完成合同审查后，可在此选择条款并生成修订合同。")
            elif not selected_revision_indices:
                st.warning("请选择至少一个“纳入合同修订”的条款。")
            else:
                revision_signature = (
                    tuple(selected_revision_indices),
                    st.session_state.final_report,
                )
                if (
                    st.session_state.revised_contract_signature
                    != revision_signature
                ):
                    with st.spinner("正在按选中条款生成修订合同..."):
                        try:
                            st.session_state.revised_contract = rewrite_contract(
                                API_BASE_URL,
                                st.session_state.clauses,
                                st.session_state.final_report,
                                selected_revision_indices,
                            )
                            st.session_state.revised_contract_signature = revision_signature
                            st.success("修订合同生成完成，未选中的条款保持原文。")
                        except Exception as exc:
                            st.error(f"生成修订合同失败: {exc}")

            if st.session_state.revised_contract:
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
    st.session_state.structured_report = None
    st.session_state.circuit_breaks = []

    status_box = st.status(f"正在初始化 ReAct 推理链路 ({target_name})...", expanded=True)
    meta_container = st.empty()
    cb_container = st.empty()
    st.caption(f"⚡ 智能体审查思考与工具调度流 [{target_name}]:")
    thought_container = st.empty()
    thought_container.markdown(render_thought_box(""), unsafe_allow_html=True)

    accumulated_tokens = ""
    extracted_final = ""
    consistency_status_recorded = False
    review_run_id = review_run_id or uuid.uuid4().hex

    try:
        for sse in stream_contract_review(
            API_BASE_URL,
            text_to_review,
            max_turns,
            review_run_id=review_run_id,
            review_side=review_side,
        ):
            event = sse["event"]
            data = sse["data"]

            if event == "start":
                status_box.update(label=f"[{data.get('task_id', target_name)}] 正在制定审查策略...")
                if data.get("trace_id"):
                    status_box.write(f"🧾 调试记录 ID：`{data['trace_id']}`（项目目录下的 data/review_logs/review_trace.jsonl）")
            elif event == "token":
                accumulated_tokens += data.get("token", "")
                thought_container.markdown(
                    render_reasoning_text(accumulated_tokens),
                    unsafe_allow_html=True,
                )
            elif event == "tool_start":
                status_box.write(f"🔧 **调度工具** `{data.get('tool')}`: 检索 *'{data.get('query')}'*")
            elif event == "tool_result":
                collect_evidence(data.get("observation", ""))
                status_box.write("📖 **已检索依据并注入模型工作记忆**")
            elif event == "validation_start":
                status_box.write("🔎 正在逐项核对原文子条款、合同事实与检索依据…")
            elif event == "validation_complete":
                status_box.write(f"✅ {data.get('message', '最终一致性检查完成')}")
                st.session_state.consistency_checks.append({"status": "passed", "message": data.get("message", "最终一致性检查完成")})
                consistency_status_recorded = True
            elif event == "validation_failed":
                status_box.write(f"⚠️ {data.get('message', '最终一致性检查未完成，保留初稿')}")
                st.session_state.consistency_checks.append({"status": "failed", "message": data.get("message", "最终一致性检查未完成，保留初稿")})
                consistency_status_recorded = True
            elif event == "circuit_break":
                st.session_state.circuit_breaks.append(data)
                recs_markdown = "\n".join([f"- **{r}**" for r in data.get("recommendations", [])])
                cb_container.markdown(f"""
                        <div class="circuit-break-card">
                            <strong>⚠️ 触发熔断保护，切入大模型知识推理</strong><br>
                            • <strong>任务</strong>: {data.get('task_id')}<br>
                            • <strong>归因</strong>: {data.get('reason_type')} ({data.get('detail')})<br>
                            • <strong>建议向知识库补充的内容</strong>:<br>
                            {recs_markdown}
                        </div>
                        """, unsafe_allow_html=True)
            elif event == "generation_meta":
                st.session_state.latest_meta = data
                meta_container.markdown(
                    f"<span class='meta-badge'>任务: {data.get('task_id')}</span>"
                    f"<span class='meta-badge'>轮次: {data.get('turn')}</span>"
                    f"<span class='meta-badge'>输入: {data.get('prompt_chars')} 字 (~{data.get('est_prompt_tokens')} Tks)</span>"
                    f"<span class='meta-badge'>输出: {data.get('output_chars')} 字 ({data.get('generated_tokens')} Tks)</span>",
                    unsafe_allow_html=True,
                )
            elif event == "error":
                status_box.update(
                    label=f"❌ 审查失败：{data.get('error', '网关返回错误')}",
                    state="error",
                    expanded=True,
                )
                st.session_state.is_reviewing = False
                return None
            elif event == "final_report":
                if not data.get("is_complete", data.get("status") == "success"):
                    status_box.update(
                        label="⚠️ 审查未完整完成，结果未保存",
                        state="error",
                        expanded=True,
                    )
                    st.session_state.is_reviewing = False
                    return None
                extracted_final = data.get("raw_report", "")
                st.session_state.structured_report = data.get("structured_report")
                if not data.get("consistency_checked") and not consistency_status_recorded:
                    st.session_state.consistency_checks.append({
                        "status": "failed",
                        "message": data.get("consistency_message", "该报告未完成最终一致性检查"),
                    })
                status_box.update(
                    label="✅ 审查完成！",
                    state="complete",
                    expanded=False,
                )
            elif event == "done":
                break

        if not extracted_final:
            status_box.update(label="⚠️ 未收到完整审查报告", state="error", expanded=True)
            st.session_state.is_reviewing = False
            return None

        st.session_state.is_reviewing = False
        return clean_report_content(extracted_final or accumulated_tokens)
    except Exception as e:
        st.error(f"连接推理网关异常: {e}")
        st.session_state.is_reviewing = False
        return None


# ==================== 7. 多线程并发调度器 ====================
def _worker_clause_review(
    clause: Dict[str, Any], turns: int, review_run_id: str, review_side: str
) -> Dict[str, Any]:
    clause_text = clause_review_text(clause)
    accumulated_tokens = ""
    extracted_final = ""
    final_complete = False
    logs = []
    evidence_records: Dict[str, Any] = {}
    consistency_checks = []
    circuit_break_info = None

    try:
        for sse in stream_contract_review(
            API_BASE_URL,
            clause_text,
            turns,
            review_run_id=review_run_id,
            review_side=review_side,
        ):
            event = sse["event"]
            data = sse["data"]
            if event == "start" and data.get("trace_id"):
                logs.append(f"调试记录 ID: {data['trace_id']}")
            elif event == "token":
                accumulated_tokens += data.get("token", "")
            elif event == "duplicate_action_reused":
                logs.append(data.get("message", "检测到重复检索，已复用原结果"))
            elif event == "circuit_break":
                circuit_break_info = data
                logs.append(f"⚠️ 触发熔断: {data.get('reason_type')}")
            elif event == "tool_start":
                logs.append(f"检索: {data.get('query')}")
            elif event == "tool_result":
                collect_evidence(data.get("observation", ""), evidence_records)
            elif event == "validation_complete":
                logs.append(data.get("message", "最终一致性检查完成"))
                consistency_checks.append({"status": "passed", "message": data.get("message", "最终一致性检查完成")})
            elif event == "validation_failed":
                logs.append(data.get("message", "最终一致性检查未完成"))
                consistency_checks.append({"status": "failed", "message": data.get("message", "最终一致性检查未完成")})
            elif event == "final_report":
                extracted_final = data.get("raw_report", "")
                final_complete = data.get("is_complete", data.get("status") == "success")
                if not data.get("consistency_checked") and not consistency_checks:
                    consistency_checks.append({
                        "status": "failed",
                        "message": data.get("consistency_message", "该报告未完成一致性检查"),
                    })
            elif event == "error":
                raise RuntimeError(data.get("error", "网关返回错误"))
            elif event == "done":
                break

        if not extracted_final or not final_complete:
            raise RuntimeError("未收到完整的最终审查报告")
        report_content = clean_report_content(extracted_final)
        return {
            "index": clause.get("index", 0),
            "title": clause.get("title", ""),
            "report": report_content,
            "logs": logs,
            "circuit_break": circuit_break_info,
            "evidence_records": evidence_records,
            "consistency_checks": consistency_checks,
            "success": bool(report_content),
        }
    except Exception as e:
        return {"index": clause.get("index", 0), "title": clause.get("title", ""), "report": f"审查异常: {e}", "logs": [str(e)], "circuit_break": None, "evidence_records": evidence_records, "consistency_checks": consistency_checks, "success": False}


def execute_concurrent_clause_review(
    clauses_to_review: List[Dict[str, Any]], max_workers: int,
    review_run_id: Optional[str] = None,
    review_side: str = "neutral",
) -> str:
    st.session_state.is_reviewing = True
    st.session_state.structured_report = None
    st.session_state.circuit_breaks = []
    st.session_state.evidence_records = {}
    st.session_state.consistency_checks = []
    review_run_id = review_run_id or uuid.uuid4().hex
    total_count = len(clauses_to_review)

    st.markdown(f"##### ⚡ 正在启用 {max_workers} 路线程并发审查 (共 {total_count} 个条款)...")
    progress_bar = st.progress(0, text=f"准备调度并发任务 (0/{total_count})...")

    status_placeholders = {}
    display_cols_count = min(4, max(2, min(total_count, max_workers)))
    st_cols = st.columns(display_cols_count, gap="small")
    for i, c in enumerate(clauses_to_review):
        col_target = st_cols[i % display_cols_count]
        with col_target:
            status_placeholders[c["index"]] = st.status(f"⏳ 排队中: {c['title']}", expanded=False)

    results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_worker_clause_review, c, max_turns, review_run_id, review_side): c["index"]
            for c in clauses_to_review
        }
        for future in as_completed(future_map):
            c_idx = future_map[future]
            res = future.result()
            results.append(res)
            completed_count += 1

            if res.get("circuit_break"):
                st.session_state.circuit_breaks.append(res["circuit_break"])
            st.session_state.evidence_records.update(res.get("evidence_records", {}))
            st.session_state.consistency_checks.extend(res.get("consistency_checks", []))

            card = status_placeholders.get(c_idx)
            if card:
                if res["success"]:
                    card.update(
                        label=f"✅ 完成: {res['title']}",
                        state="complete",
                        expanded=False,
                    )
                    for l in res["logs"][-3:]:
                        card.write(f"- {l}")
                else:
                    card.update(label=f"❌ 失败: {res['title']}", state="error", expanded=True)

            progress_bar.progress(completed_count / total_count, text=f"并发审查进度 ({completed_count}/{total_count})...")

    progress_bar.empty()
    results.sort(key=lambda x: x["index"])
    aggregated_reports = [f"### 条款 {r['index']}: {r['title']}\n\n{r['report']}\n\n---" for r in results]

    st.session_state.is_reviewing = False
    return "# 综合合同审查终审报告 (多路并发精审汇总)\n\n" + "\n\n".join(aggregated_reports)


# ==================== 8. 下部：审查轨迹与展示 ====================
with st.container():
    st.markdown("#### 🔍 审查推理轨迹与法务终审报告")

    if selected_clause_idx is not None:
        target_clause = next((c for c in st.session_state.clauses if c.get("index") == selected_clause_idx), None)
        if target_clause:
            st.session_state.evidence_records = {}
            st.session_state.consistency_checks = []
            target_text = clause_review_text(target_clause)
            report = execute_stream_review(
                target_text,
                target_name=target_clause["title"],
                review_side=review_side,
            )
            if report:
                st.session_state.final_report = f"## 针对【{target_clause['title']}】的专属审查意见\n\n" + report
                st.rerun()

    elif start_full_review:
        st.session_state.evidence_records = {}
        st.session_state.consistency_checks = []
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

    # 渲染前端收集到的所有熔断告警与知识库增补清单
    if st.session_state.circuit_breaks:
        with st.expander(f"⚠️ 审查过程中触发了 {len(st.session_state.circuit_breaks)} 处熔断保护 (点击查看知识库增补建议)", expanded=True):
            for cb in st.session_state.circuit_breaks:
                recs_md = "\n".join([f"- **{r}**" for r in cb.get("recommendations", [])])
                st.markdown(f"""
                <div class="circuit-break-card">
                    <strong>📌 任务单元: {cb.get('task_id')}</strong><br>
                    • <strong>熔断归因</strong>: {cb.get('reason_type')}<br>
                    • <strong>触发详情</strong>: {cb.get('detail')}<br>
                    • <strong>尝试过的检索词</strong>: <code>{', '.join(cb.get('attempted_queries', []))}</code><br>
                    • <strong>建议向知识库补充的资料</strong>:<br>
                    {recs_md}
                </div>
                """, unsafe_allow_html=True)

    # 渲染 Markdown 报告
    if st.session_state.final_report:
        check_results = st.session_state.consistency_checks
        failed_checks = [item for item in check_results if item.get("status") != "passed"]
        if check_results and failed_checks:
            st.warning(f"最终一致性检查未完成或未通过：{len(failed_checks)}/{len(check_results)} 个条款。初稿可能仍有事实冲突或依据错配，请优先复核。")
        elif check_results:
            st.info(f"已执行 {len(check_results)} 个条款的一致性复核；引用依据仍请结合原文核对。")
        evidence_records = st.session_state.evidence_records
        if evidence_records:
            with st.expander(f"📚 本次检索依据（{len(evidence_records)} 条，点击查看命中原文）", expanded=True):
                for evidence_id, evidence in evidence_records.items():
                    citation = f"《{evidence.get('doc_name', '参考文档')}》"
                    if evidence.get("article_no"):
                        citation += f" {evidence['article_no']}"
                    if evidence.get("title"):
                        citation += f" {evidence['title']}"
                    citation += f" · 第 {evidence.get('page_start', '?')} 页"
                    if st.button(f"查看原文：{citation}", key=f"all_evidence_{evidence_id}"):
                        show_evidence_content(evidence)
        else:
            st.info("本次审查没有收集到法规原文证据。若报告列出了法条依据，请确认法规检索工具已成功返回结果后重新审查。")
        if st.session_state.structured_report is None:
            parsed_report = parse_structured_report(st.session_state.final_report)
            if parsed_report:
                st.session_state.structured_report = parsed_report.model_dump(mode="json")
        structured_reviews = (st.session_state.structured_report or {}).get("reviews", [])
        if structured_reviews:
            st.caption(
                "按法律效力、商业后果、救济成本及所选审查立场综合分级；纯执行能力要求归为提示。"
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
                        st.markdown(f"#### {index}. {item.get('clause_topic', '未命名条款')}")
                        if item.get("risk_type"):
                            st.markdown(f"**风险类型：** {item['risk_type']}")
                        st.markdown(f"**风险等级：** {label}")
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
                        visible_basis = format_report_evidence_refs(
                            legal_basis, st.session_state.evidence_records
                        ).strip()
                        visible_basis = re.sub(r"^[\s、，,；;]+|[\s、，,；;]+$", "", visible_basis)
                        if visible_basis:
                            st.markdown(f"**法律/合规依据：** {visible_basis}")
                        elif evidence_ids:
                            st.markdown("**法律/合规依据：** 关联以下检索原文（请核对原文是否支持本项分析）")
                        else:
                            st.markdown("**法律/合规依据：** 未检索到直接依据")
                        for evidence_id in evidence_ids:
                            evidence = st.session_state.evidence_records.get(evidence_id)
                            if evidence:
                                citation = f"《{evidence.get('doc_name', '参考文档')}》 {evidence.get('article_no') or ''} {evidence.get('title') or ''}".strip()
                                if st.button(f"查看依据原文：{citation}", key=f"evidence_{level}_{index}_{evidence_id}"):
                                    show_evidence_content(evidence)
                            else:
                                st.warning(f"报告引用的证据 {evidence_id} 未在本次检索记录中找到。")
                        if not evidence_ids and evidence_records:
                            article_refs = set(re.findall(r"第\s*[零〇一二三四五六七八九十百千万两0-9]+\s*条", legal_basis))
                            related = [
                                evidence for evidence in evidence_records.values()
                                if evidence.get("article_no") in article_refs
                            ]
                            for evidence_index, evidence in enumerate(related):
                                citation = f"{evidence.get('doc_name', '参考文档')} {evidence.get('article_no', '')} {evidence.get('title', '')}".strip()
                                if st.button(f"查看命中原文：{citation}", key=f"basis_{level}_{index}_{evidence_index}"):
                                    show_evidence_content(evidence)
                        st.markdown(f"**风险分析：** {item.get('issue', '')}")
                        st.markdown(f"**修改建议：** {item.get('suggested_revision', '')}")
                        if index < len(items):
                            st.divider()
            with st.expander("查看完整原始审查报告"):
                st.markdown(format_report_evidence_refs(
                    clean_report_content(st.session_state.final_report),
                    st.session_state.evidence_records,
                ))
        else:
            st.warning("报告未能解析成风险条目，以下显示完整原始报告。")
            st.markdown(format_report_evidence_refs(
                clean_report_content(st.session_state.final_report),
                st.session_state.evidence_records,
            ))

        st.markdown("<br>", unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        export_filename = f"contract_review_{int(time.time())}"
        readable_report = format_report_evidence_refs(
            st.session_state.final_report, st.session_state.evidence_records
        )
        with export_col1:
            st.download_button(
                label="📥 导出 Markdown 报告",
                data=readable_report,
                file_name=f"{export_filename}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with export_col2:
            try:
                docx_report = report_to_docx(readable_report)
                st.download_button(
                    label="📄 导出 Word 报告",
                    data=docx_report,
                    file_name=f"{export_filename}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except RuntimeError as exc:
                st.error(str(exc))
    elif not st.session_state.is_reviewing:
        st.info("💡 操作指引：确认待审合同后，点击【开始执行合同智能合规审查】或展开条款明细点击【单独审查】。")
