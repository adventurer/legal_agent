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


def _review_for_clause(
    review_report: str, clause_index: int, clause_title: str
) -> str:
    """Keep each rewrite request focused on the matching review section."""
    report = (review_report or "").strip()
    if not report:
        return ""

    markers = [
        f"条款 {clause_index}",
        f"条款{clause_index}",
        clause_title.strip(),
    ]
    start = next(
        (report.find(marker) for marker in markers if marker and report.find(marker) >= 0),
        -1,
    )
    if start < 0:
        return report[:4000]

    next_section = re.search(
        r"\n\s*#{1,6}\s+条款\s*\d+",
        report[start + 1 :],
        flags=re.IGNORECASE,
    )
    end = start + 1 + next_section.start() if next_section else len(report)
    return report[start:end].strip()[:6000]


def _rewrite_clause(
    agent: Any,
    clause: Dict[str, Any],
    review_report: str,
) -> Dict[str, Any]:
    """Rewrite one clause in an isolated model request."""
    index = int(clause.get("index", 0))
    title = str(clause.get("title", f"条款 {index}"))
    clause_review = _review_for_clause(review_report, index, title)
    prompt = CONTRACT_REWRITE_PROMPT.format(
        clauses=json.dumps([clause], ensure_ascii=False),
        review_report=clause_review,
    )
    response = agent.client.chat.completions.create(
        model=agent.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
        stream=False,
    )
    result = _extract_json(response.choices[0].message.content)
    revised = next(
        (
            item
            for item in result["revised_clauses"]
            if int(item.get("index", -1)) == index and item.get("revised_text")
        ),
        None,
    )
    if revised is None:
        raise ValueError(f"模型未返回条款 {index} 的有效修订结果")
    return revised


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

    revised_by_index = {
        int(clause["index"]): _rewrite_clause(agent, clause, review_report)
        for clause in selected_clauses
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
