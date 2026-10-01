"""Streamlit page for managing legal references and review rules."""

from typing import Any, Dict, List, Optional

import httpx
import streamlit as st


REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=240.0, write=240.0, pool=10.0)
CATEGORY_LABELS = {
    "law": "法典与法规",
    "enterprise": "企业知识库",
    "general": "其他资料",
}
RISK_LEVELS = ["High", "Medium", "Low", "Notice"]


def _request(
    method: str,
    api_base_url: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        response = httpx.request(
            method,
            f"{api_base_url}{path}",
            params=params,
            json=json,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        detail = str(exc)
        management_route_missing = False
        if exc.response is not None:
            management_route_missing = (
                exc.response.status_code == 404
                and path.startswith("/api/v1/knowledge/")
            )
            if management_route_missing:
                st.session_state["knowledge_gateway_restart_required"] = True
            try:
                detail = exc.response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
        if not management_route_missing:
            st.error(f"请求失败：{detail}")
    return None


def _request_document_preview(
    api_base_url: str,
    filename: str,
) -> Optional[httpx.Response]:
    path = "/api/v1/knowledge/document-content"
    try:
        response = httpx.get(
            f"{api_base_url}{path}",
            params={"filename": filename},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response
    except httpx.HTTPError as exc:
        detail = str(exc)
        if exc.response is not None:
            if exc.response.status_code == 404:
                try:
                    detail = exc.response.json().get("detail", detail)
                except (ValueError, AttributeError):
                    pass
            if exc.response.status_code == 404 and detail == "Not Found":
                st.session_state["knowledge_gateway_restart_required"] = True
                st.warning("当前网关尚未加载资料查看接口，请在合同审查工作台重启网关后重试。")
                return None
        st.error(f"资料读取失败：{detail}")
    return None


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _render_document_library(
    api_base_url: str,
    category: str,
    documents: List[Dict[str, Any]],
) -> None:
    st.subheader(CATEGORY_LABELS[category])
    with st.form(f"knowledge-upload-{category}"):
        uploaded_file = st.file_uploader(
            "上传 PDF、TXT 或 Markdown",
            type=["pdf", "txt", "md"],
            max_upload_size=20,
            key=f"knowledge-file-{category}",
        )
        submitted = st.form_submit_button("导入并建立索引", type="primary")
    if submitted:
        if uploaded_file is None:
            st.warning("请先选择文件。")
        else:
            result = _request(
                "POST",
                api_base_url,
                "/api/v1/knowledge/documents",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                data={"category": category},
            )
            if result is not None:
                st.success(f"已导入：{result['filename']}，当前索引 {result['indexed_pages']} 条。")
                st.rerun()

    visible_documents = [item for item in documents if item["category"] == category]
    if visible_documents:
        filenames = [item["filename"] for item in visible_documents]
        preview_key = f"knowledge-preview-{category}"
        for index, item in enumerate(visible_documents):
            name_column, action_column = st.columns([6, 1])
            name_column.markdown(f"**{item['filename']}**")
            name_column.caption(
                f"{item['tag']} · {item['indexed_segments']} 条 · "
                f"{_format_size(item['size_bytes'])}"
            )
            if action_column.button(
                "查看",
                icon=":material/visibility:",
                key=f"knowledge-view-{category}-{index}",
            ):
                st.session_state[preview_key] = item["filename"]

        selected_preview = st.session_state.get(preview_key)
        if selected_preview not in filenames:
            st.session_state.pop(preview_key, None)
            selected_preview = None
        if selected_preview:
            preview_response = _request_document_preview(api_base_url, selected_preview)
            if preview_response is not None:
                with st.expander(f"资料预览：{selected_preview}", expanded=True):
                    if selected_preview.lower().endswith(".pdf"):
                        st.pdf(preview_response.content)
                    else:
                        language = "markdown" if selected_preview.lower().endswith(".md") else None
                        content = preview_response.content.decode("utf-8-sig", errors="replace")
                        st.code(content, language=language, wrap_lines=True)

        selected_filename = st.selectbox(
            "选择要移除的资料",
            filenames,
            key=f"knowledge-delete-select-{category}",
        )
        confirmed = st.checkbox(
            "确认删除资料及其索引缓存",
            key=f"knowledge-delete-confirm-{category}",
        )
        if st.button(
            "删除资料",
            disabled=not confirmed,
            key=f"knowledge-delete-{category}",
        ):
            result = _request(
                "DELETE",
                api_base_url,
                "/api/v1/knowledge/documents",
                params={"filename": selected_filename},
            )
            if result is not None:
                st.toast("资料已删除，检索索引已刷新。")
                st.rerun()
    else:
        st.caption("当前分类暂无资料。")


def _render_rule_book(api_base_url: str, rules: List[Dict[str, Any]]) -> None:
    st.subheader("自编法典")
    if rules:
        st.dataframe(
            [
                {
                    "主题": rule["topic"],
                    "风险等级": rule["risk_level"],
                    "关键词": rule["keywords"],
                    "标准要求": rule["standard_requirement"],
                }
                for rule in rules
            ],
            use_container_width=True,
            hide_index=True,
        )

    rule_options = ["新建规则"] + [rule["topic"] for rule in rules]
    selected_topic = st.selectbox("编辑规则", rule_options, key="knowledge-rule-selection")
    current = next((rule for rule in rules if rule["topic"] == selected_topic), {})
    form_key = f"knowledge-rule-form-{selected_topic}"
    with st.form(form_key):
        topic = st.text_input("主题", value=current.get("topic", ""), key=f"rule-topic-{selected_topic}")
        keywords = st.text_input(
            "检索关键词（逗号分隔）",
            value=current.get("keywords", ""),
            key=f"rule-keywords-{selected_topic}",
        )
        risk_level = st.selectbox(
            "风险等级",
            RISK_LEVELS,
            index=RISK_LEVELS.index(current["risk_level"])
            if current.get("risk_level") in RISK_LEVELS else 1,
            key=f"rule-risk-{selected_topic}",
        )
        standard_requirement = st.text_area(
            "审查标准",
            value=current.get("standard_requirement", ""),
            height=130,
            key=f"rule-standard-{selected_topic}",
        )
        forbidden_pattern = st.text_area(
            "禁止情形",
            value=current.get("forbidden_pattern", ""),
            height=90,
            key=f"rule-forbidden-{selected_topic}",
        )
        recommended_clause = st.text_area(
            "建议条款",
            value=current.get("recommended_clause", ""),
            height=110,
            key=f"rule-clause-{selected_topic}",
        )
        submitted = st.form_submit_button("保存规则", type="primary")

    if submitted:
        result = _request(
            "PUT",
            api_base_url,
            "/api/v1/knowledge/rules",
            json={
                "topic": topic,
                "keywords": keywords,
                "risk_level": risk_level,
                "standard_requirement": standard_requirement,
                "forbidden_pattern": forbidden_pattern,
                "recommended_clause": recommended_clause,
            },
        )
        if result is not None:
            st.toast("规则已保存。")
            st.rerun()

    if current:
        confirmed = st.checkbox("确认删除此规则", key=f"rule-delete-confirm-{selected_topic}")
        if st.button(
            "删除规则",
            disabled=not confirmed,
            key=f"rule-delete-{selected_topic}",
        ):
            result = _request(
                "DELETE",
                api_base_url,
                "/api/v1/knowledge/rules",
                params={"topic": selected_topic},
            )
            if result is not None:
                st.toast("规则已删除。")
                st.rerun()


def render_knowledge_management_page(
    api_base_url: str,
    gateway_healthy: bool,
) -> None:
    st.title("知识库管理")
    if not gateway_healthy:
        st.warning("网关未启动，知识库暂不可管理。")
        return

    st.session_state["knowledge_gateway_restart_required"] = False
    documents_result = _request("GET", api_base_url, "/api/v1/knowledge/documents")
    rules_result = _request("GET", api_base_url, "/api/v1/knowledge/rules")
    if documents_result is None or rules_result is None:
        if st.session_state.get("knowledge_gateway_restart_required"):
            st.warning("当前网关尚未加载知识库管理接口。请返回合同审查工作台，在服务管理中重启网关。")
        return

    documents = documents_result.get("documents", [])
    rules = rules_result.get("rules", [])
    counts = {
        category: sum(item["category"] == category for item in documents)
        for category in CATEGORY_LABELS
    }
    metric_columns = st.columns(4)
    metric_columns[0].metric("法典与法规", counts["law"])
    metric_columns[1].metric("企业知识库", counts["enterprise"])
    metric_columns[2].metric("其他资料", counts["general"])
    metric_columns[3].metric("自编规则", len(rules))

    law_tab, enterprise_tab, general_tab, rules_tab = st.tabs(
        ["法典与法规", "企业知识库", "其他资料", "自编法典"]
    )
    with law_tab:
        _render_document_library(api_base_url, "law", documents)
    with enterprise_tab:
        _render_document_library(api_base_url, "enterprise", documents)
    with general_tab:
        _render_document_library(api_base_url, "general", documents)
    with rules_tab:
        _render_rule_book(api_base_url, rules)
