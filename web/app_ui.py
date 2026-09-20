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
import html
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from services.report_parser import clean_report_content
from web.api_client import check_gateway_health as fetch_gateway_health
from web.api_client import upload_contract_file as send_contract_file
from web.report_exporter import report_to_docx
from web.sse_client import stream_contract_review

API_BASE_URL = "http://127.0.0.1:9000"

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
if "is_reviewing" not in st.session_state:
    st.session_state.is_reviewing = False
if "truncation_warning" not in st.session_state:
    st.session_state.truncation_warning = None
if "latest_meta" not in st.session_state:
    st.session_state.latest_meta = None
if "circuit_breaks" not in st.session_state:
    st.session_state.circuit_breaks = []  # 存储各任务熔断详情


# ==================== 3. 辅助函数 ====================
def check_gateway_health() -> Dict[str, Any]:
    try:
        return fetch_gateway_health(API_BASE_URL)
    except Exception:
        return {"status": "unreachable"}


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


# ==================== 4. 顶部控制栏 ====================
header_col, status_col = st.columns([3, 1])
with header_col:
    st.markdown('<div class="main-header">⚖️ 本地法务合同审查 Agent 工作台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">全宽自适应可视 · 层次化条款切片 · 思考流追踪 · 结构化审查报告</div>', unsafe_allow_html=True)

with status_col:
    health = check_gateway_health()
    if health.get("status") == "healthy":
        st.success(f"🟢 服务就绪 ({health.get('model', 'qwen2.5')})")
    else:
        st.error("🔴 网关未启动 (请运行 api_server.py)")

with st.container():
    st.markdown('<div class="top-control-panel">', unsafe_allow_html=True)
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.1, 0.9, 0.9], gap="medium")

    with ctrl_col1:
        review_mode = st.radio(
            "审查策略：",
            ["分条款并发精审 (推荐·高效防截断)", "全篇宏观快审 (整篇审查)"],
            horizontal=False,
            index=0,
        )

    with ctrl_col2:
        concurrency = st.slider("⚡ 并发线程数", min_value=1, max_value=100, value=8, step=1)

    with ctrl_col3:
        max_turns = st.slider("每轮最大探索步数", min_value=3, max_value=8, value=5, step=1)
        debug_mode = st.checkbox(
            "开启模型交互 Debug",
            value=False,
            help="开启后，API 网关控制台会打印每轮模型请求、响应和工具结果。",
        )

    with ctrl_col4:
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
            st.session_state.circuit_breaks = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ==================== 5. 中部工作区 ====================
with st.container():
    st.markdown("#### 📄 待审查合同数据源")
    top_col1, top_col2 = st.columns([1, 2], gap="medium")
    with top_col1:
        uploaded_file = st.file_uploader(
            "上传合同文件 (Word / PDF / 图片 / TXT)",
            type=["docx", "doc", "pdf", "png", "jpg", "jpeg", "webp", "txt"],
        )
        if uploaded_file is not None and st.session_state.get("last_uploaded_name") != uploaded_file.name:
            with st.spinner("正在执行多模态解析与条款切分..."):
                res = upload_contract_file(uploaded_file, st.session_state.session_id)
                if res and res.get("code") == 200:
                    data = res["data"]
                    st.session_state.session_id = data["session_id"]
                    st.session_state.clauses = data["clauses"]
                    st.session_state.full_contract_text = "\n\n".join([f"{c['title']}\n{c['content']}" for c in data["clauses"]])
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.session_state.final_report = ""
                    st.session_state.circuit_breaks = []
                    st.success(f"解析成功，切分出 {len(data['clauses'])} 个条款单元！")
                    st.rerun()

    with top_col2:
        st.text_area("合同原文预览（只读）：", value=st.session_state.full_contract_text, height=130, disabled=True)

    selected_clause_idx = None
    if st.session_state.clauses:
        with st.expander(f"📑 条款结构化明细表 (共 {len(st.session_state.clauses)} 个单元)", expanded=False):
            clause_cols = st.columns(2, gap="medium")
            for idx, c in enumerate(st.session_state.clauses):
                col_target = clause_cols[idx % 2]
                with col_target:
                    c_col1, c_col2 = st.columns([4, 1])
                    with c_col1:
                        st.markdown(f"**`#{c.get('index')}` {c.get('title')}**")
                    with c_col2:
                        if st.button("单独审查", key=f"btn_single_{c.get('index')}"):
                            selected_clause_idx = c.get("index")
                    st.caption(c.get("content", "")[:120] + "...")
                    st.divider()

    start_full_review = st.button("🚀 开始执行合同智能合规审查", type="primary", use_container_width=True, disabled=st.session_state.is_reviewing or not bool(st.session_state.full_contract_text))

st.markdown("<hr style='margin: 1.4rem 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)


# ==================== 6. 单通道 SSE 审查逻辑 ====================
def execute_stream_review(
    text_to_review: str,
    target_name: str = "合同正文",
    debug: bool = False,
) -> Optional[str]:
    st.session_state.is_reviewing = True
    st.session_state.circuit_breaks = []

    status_box = st.status(f"正在初始化 ReAct 推理链路 ({target_name})...", expanded=True)
    meta_container = st.empty()
    cb_container = st.empty()
    st.caption(f"⚡ 智能体审查思考与工具调度流 [{target_name}]:")
    thought_container = st.empty()
    thought_container.markdown(render_thought_box(""), unsafe_allow_html=True)

    accumulated_tokens = ""
    extracted_final = ""

    try:
        for sse in stream_contract_review(
            API_BASE_URL, text_to_review, max_turns, debug=debug
        ):
            event = sse["event"]
            data = sse["data"]

            if event == "start":
                status_box.update(label=f"[{data.get('task_id', target_name)}] 正在制定审查策略...")
            elif event == "token":
                accumulated_tokens += data.get("token", "")
                thought_container.markdown(
                    render_reasoning_text(accumulated_tokens),
                    unsafe_allow_html=True,
                )
            elif event == "tool_start":
                status_box.write(f"🔧 **调度工具** `{data.get('tool')}`: 检索 *'{data.get('query')}'*")
            elif event == "tool_result":
                status_box.write("📖 **已检索依据并注入模型工作记忆**")
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
            elif event == "final_report":
                extracted_final = data.get("raw_report", "")
                status_box.update(
                    label="✅ 审查完成！",
                    state="complete",
                    expanded=False,
                )
            elif event == "done":
                break

        if not extracted_final and "Final:" in accumulated_tokens:
            extracted_final = accumulated_tokens.split("Final:", 1)[1].strip()

        st.session_state.is_reviewing = False
        return clean_report_content(extracted_final or accumulated_tokens)
    except Exception as e:
        st.error(f"连接推理网关异常: {e}")
        st.session_state.is_reviewing = False
        return None


# ==================== 7. 多线程并发调度器 ====================
def _worker_clause_review(
    clause: Dict[str, Any], turns: int, debug: bool = False
) -> Dict[str, Any]:
    clause_text = f"{clause['title']}\n{clause['content']}"
    accumulated_tokens = ""
    extracted_final = ""
    logs = []
    circuit_break_info = None

    try:
        for sse in stream_contract_review(
            API_BASE_URL, clause_text, turns, debug=debug
        ):
            event = sse["event"]
            data = sse["data"]
            if event == "token":
                accumulated_tokens += data.get("token", "")
            elif event == "circuit_break":
                circuit_break_info = data
                logs.append(f"⚠️ 触发熔断: {data.get('reason_type')}")
            elif event == "tool_start":
                logs.append(f"检索: {data.get('query')}")
            elif event == "final_report":
                extracted_final = data.get("raw_report", "")
            elif event == "done":
                break

        report_content = clean_report_content(
            extracted_final
            or (
                accumulated_tokens.split("Final:", 1)[-1].strip()
                if "Final:" in accumulated_tokens
                else accumulated_tokens
            )
        )
        return {
            "index": clause.get("index", 0),
            "title": clause.get("title", ""),
            "report": report_content,
            "logs": logs,
            "circuit_break": circuit_break_info,
            "success": True,
        }
    except Exception as e:
        return {"index": clause.get("index", 0), "title": clause.get("title", ""), "report": f"审查异常: {e}", "logs": [str(e)], "circuit_break": None, "success": False}


def execute_concurrent_clause_review(
    clauses_to_review: List[Dict[str, Any]], max_workers: int, debug: bool = False
) -> str:
    st.session_state.is_reviewing = True
    st.session_state.circuit_breaks = []
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
            executor.submit(_worker_clause_review, c, max_turns, debug): c["index"]
            for c in clauses_to_review
        }
        for future in as_completed(future_map):
            c_idx = future_map[future]
            res = future.result()
            results.append(res)
            completed_count += 1

            if res.get("circuit_break"):
                st.session_state.circuit_breaks.append(res["circuit_break"])

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
            target_text = f"{target_clause['title']}\n{target_clause['content']}"
            report = execute_stream_review(
                target_text, target_name=target_clause["title"], debug=debug_mode
            )
            if report:
                st.session_state.final_report = f"## 针对【{target_clause['title']}】的专属审查意见\n\n" + report
                st.rerun()

    elif start_full_review:
        if not st.session_state.full_contract_text.strip():
            st.warning("请先上传合同文件或载入样例数据！")
        else:
            if "并发精审" in review_mode and st.session_state.clauses:
                article_clauses = [c for c in st.session_state.clauses if c.get("type") == "article"] or st.session_state.clauses
                if concurrency == 1:
                    total_articles = len(article_clauses)
                    aggregated_reports = []
                    progress_bar = st.progress(0, text="单路条款逐条精审中...")
                    for idx, clause in enumerate(article_clauses):
                        progress_bar.progress((idx + 1) / total_articles, text=f"正在精审 ({idx+1}/{total_articles}): {clause['title']}")
                        report_part = execute_stream_review(
                            f"{clause['title']}\n{clause['content']}",
                            target_name=clause["title"],
                            debug=debug_mode,
                        )
                        if report_part:
                            aggregated_reports.append(f"### 条款 {clause['index']}: {clause['title']}\n\n{report_part}\n\n---")
                    progress_bar.empty()
                    st.session_state.final_report = "# 综合合同审查终审报告 (逐条精审汇总)\n\n" + "\n\n".join(aggregated_reports)
                    st.rerun()
                else:
                    final_aggregated = execute_concurrent_clause_review(
                        article_clauses, max_workers=concurrency, debug=debug_mode
                    )
                    st.session_state.final_report = final_aggregated
                    st.rerun()
            else:
                report = execute_stream_review(
                    st.session_state.full_contract_text,
                    target_name="全篇合同全文",
                    debug=debug_mode,
                )
                if report:
                    st.session_state.final_report = report
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
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(clean_report_content(st.session_state.final_report))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        export_filename = f"contract_review_{int(time.time())}"
        with export_col1:
            st.download_button(
                label="📥 导出 Markdown 报告",
                data=st.session_state.final_report,
                file_name=f"{export_filename}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with export_col2:
            try:
                docx_report = report_to_docx(st.session_state.final_report)
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