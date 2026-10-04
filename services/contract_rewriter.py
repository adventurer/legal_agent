"""按用户确认的审查意见进行条款级合同重写。"""

import json
import re
from typing import Any, Dict, List

from core.prompts import CONTRACT_REWRITE_PROMPT


_SUBCLAUSE_MARKER_PATTERN = re.compile(
    r"^\s*(?P<marker>(?:\d+(?:\.\d+)*[、.．)]?|[（(]\d+[）)]|"
    r"[（(][一二三四五六七八九十百千万]+[）)]|"
    r"[一二三四五六七八九十百千万]+[、.．）)]))\s*(?=\S)",
    re.MULTILINE,
)


def _normalize_clause_title(value: Any) -> str:
    """Normalize Markdown formatting and spacing before comparing titles."""
    text = str(value or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[\]\([^)]+\)", "", text)
    text = re.sub(r"[*_`#]", "", text)
    return re.sub(r"\s+", "", text).strip().lower()


def _subclause_markers(text: str) -> List[str]:
    return [
        re.sub(r"\s+", "", match.group("marker"))
        for match in _SUBCLAUSE_MARKER_PATTERN.finditer(text)
    ]


def _preserves_subclause_markers(original_text: str, revised_text: str) -> bool:
    actual_markers = iter(_subclause_markers(revised_text))
    return all(
        any(actual == expected for actual in actual_markers)
        for expected in _subclause_markers(original_text)
    )


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

    normalized_title = _normalize_clause_title(clause_title)
    title_start = -1
    for match in re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?.+$", report):
        candidate = match.group(0)
        if normalized_title and (
            normalized_title in _normalize_clause_title(candidate)
            or _normalize_clause_title(candidate) in normalized_title
        ):
            title_start = match.start()
            break
    markers = [f"条款 {clause_index}", f"条款{clause_index}", clause_title.strip()]
    start = title_start if title_start >= 0 else next(
        (report.find(marker) for marker in markers if marker and report.find(marker) >= 0),
        -1,
    )
    if start < 0:
        return report[:4000]

    end = len(report)
    for heading in re.finditer(
        r"(?im)^\s*#{1,6}\s+(.+)$", report[start + 1 :]
    ):
        if heading.start() == 0:
            continue
        heading_text = heading.group(1)
        if re.search(
            r"(?:条款\s*\d+|第[一二三四五六七八九十百千零0-9]+条)",
            heading_text,
            flags=re.IGNORECASE,
        ):
            end = start + 1 + heading.start()
            break
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
        clause_index=index,
    )
    original_text = str(clause.get("content", ""))
    original_markers = _subclause_markers(original_text)
    normalized_title = _normalize_clause_title(title)
    retry_instruction = ""

    for attempt in range(2):
        response = agent.client.chat.completions.create(
            model=agent.model_name,
            messages=[{"role": "user", "content": prompt + retry_instruction}],
            temperature=0.0,
            max_tokens=2048,
            stream=False,
        )
        result = _extract_json(response.choices[0].message.content)
        revised = None
        for item in result["revised_clauses"]:
            if not isinstance(item, dict) or not item.get("revised_text"):
                continue
            try:
                item_index = int(item.get("index", -1))
            except (TypeError, ValueError):
                item_index = -1
            item_title = _normalize_clause_title(item.get("title", ""))
            # An explicit index is authoritative. A title match is only a fallback
            # for responses that omit the index; never let a title override it.
            if (item_index == index) or (
                item_index < 0
                and item_title
                and item_title == normalized_title
            ):
                revised = item
                break
        if revised is None:
            raise ValueError(f"模型未返回条款 {index} 的有效修订结果")

        revised_text = str(revised["revised_text"])
        if _preserves_subclause_markers(original_text, revised_text):
            return revised
        revised_markers = _subclause_markers(revised_text)
        if not attempt:
            retry_instruction = (
                "\n\n上一次修订遗漏或重排了子条款编号。"
                f"原文编号顺序：{'、'.join(original_markers)}；"
                f"上次输出编号顺序：{'、'.join(revised_markers)}。"
                "请重新输出完整条款，保留所有子条款编号、顺序及正文内容。"
            )
            continue
        raise ValueError(
            f"模型两次修订均遗漏或重排了条款 {index} 的子条款；"
            f"原文编号：{'、'.join(original_markers)}；"
            f"最后输出编号：{'、'.join(revised_markers)}"
        )
    raise ValueError(f"模型未能完成条款 {index} 的修订")


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
        if revised:
            text = str(revised["revised_text"])
            contract_parts.append(f"{title}\n{text}".strip())
        else:
            text = str(clause.get("content", ""))
            # Keep the exact extracted clause text where available (not an
            # invented title/content reconstruction for preambles or sign pages).
            contract_parts.append(str(clause.get("raw_text") or f"{title}\n{text}").strip())
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
