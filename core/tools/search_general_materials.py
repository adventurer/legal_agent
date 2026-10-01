"""Search general, non-legal business materials."""

from typing import Any, Callable, Dict, List

from .common import serialize_evidence


def search_general_materials(
    query: str,
    search_documents: Callable[[str, str, str], List[Dict[str, Any]]],
) -> str:
    evidence = search_documents(query, "通用", "general_document")
    return serialize_evidence(
        evidence,
        f"未检索到与【{query}】相关的通用资料。",
    )