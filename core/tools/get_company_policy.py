"""Search enterprise documents and internal policy rules."""

from typing import Any, Callable, Dict, List

from .common import serialize_evidence


def get_company_policy(
    query: str,
    search_documents: Callable[[str, str, str], List[Dict[str, Any]]],
    search_rules: Callable[[str], List[Dict[str, Any]]],
) -> str:
    evidence = search_documents(query, "合规", "enterprise_document")
    evidence.extend(search_rules(query))
    return serialize_evidence(
        evidence,
        f"未检索到与【{query}】相关的企业规则或合规资料。",
    )