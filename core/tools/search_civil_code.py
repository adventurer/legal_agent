"""Search authoritative law materials."""

from typing import Any, Callable, Dict, List

from .common import serialize_evidence


def search_civil_code(
    query: str,
    search_documents: Callable[[str, str, str], List[Dict[str, Any]]],
) -> str:
    evidence = search_documents(query, "法", "law")
    return serialize_evidence(
        evidence,
        f"未检索到与【{query}】相关的法规条文。",
    )