"""按用户确认的审查意见进行条款级合同重写。"""

import json
import re
from typing import Any, Dict, List

from core.prompts import CONTRACT_REWRITE_PROMPT


def _extract_json(raw_text: str) -> Dict[str, Any]:
    content = (raw_text or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("模型未返回有效的合同修订 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("revised_clauses"), list):
        raise ValueError("合同修订结果缺少 revised_clauses 列表")
    return value


def rewrite_selected_clauses(
    agent: Any,
    clauses: List[Dict[str, Any]],
    review_report: str,
    selected_indices: List[int],
) -> Dict[str, Any]:
    """只重写选中的条款，并按原条款顺序拼装完整合同。"""
    selected = set(selected_indices)
    selected_clauses = [
        clause for clause in clauses if int(clause.get("index", 0)) in selected
    ]
    if not selected_clauses:
        raise ValueError("至少选择一个需要修订的条款")

    prompt = CONTRACT_REWRITE_PROMPT.format(
        clauses=json.dumps(selected_clauses, ensure_ascii=False),
        review_report=review_report,
    )
    response = agent.client.chat.completions.create(
        model=agent.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
        stream=False,
    )
    raw_content = response.choices[0].message.content
    result = _extract_json(raw_content)
    revised_by_index = {
        int(item["index"]): item
        for item in result["revised_clauses"]
        if "index" in item and item.get("revised_text")
    }

    contract_parts = []
    revised_clauses = []
    for clause in clauses:
        index = int(clause.get("index", 0))
        title = str(clause.get("title", f"条款 {index}"))
        revised = revised_by_index.get(index)
        text = str(revised["revised_text"]) if revised else str(clause.get("content", ""))
        contract_parts.append(f"{title}\n{text}".strip())
        if revised:
            revised_clauses.append({
                "index": index,
                "title": title,
                "original_text": str(clause.get("content", "")),
                "revised_text": text,
                "change_reason": str(revised.get("change_reason", "")),
            })
    return {
        "contract_text": "\n\n".join(part for part in contract_parts if part),
        "revised_clauses": revised_clauses,
    }
