"""Search saved historical contract-review rules."""

import json
from typing import Any, Callable, Dict, List


def get_past_review_rules(
    query: str,
    search_rules: Callable[[str], List[Dict[str, Any]]],
) -> str:
    evidence = search_rules(query)
    return json.dumps({
        "evidence": evidence,
        "message": "" if evidence else f"未检索到针对【{query}】的企业审查规则。",
    }, ensure_ascii=False)