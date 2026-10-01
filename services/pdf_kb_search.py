"""Keyword scoring and result formatting for PDF knowledge pages."""

import json
import re
from typing import Any, Dict, List, Optional


_QUERY_STOPWORDS = {
    "中国", "法律", "法规", "法律法规", "合同法", "法条", "相关", "现行", "依据",
    "规定", "检索", "查询", "合同", "条款",
}


def search_pages(
    pages: List[Dict[str, Any]], query: str, filter_tag: Optional[str] = None,
    top_k: int = 2, max_snippet_len: int = 350,
) -> str:
    if not query or not query.strip():
        return json.dumps({"evidence": [], "message": "未提供有效的查询关键词。"}, ensure_ascii=False)
    if not pages:
        return json.dumps({"evidence": [], "message": f"未找到任何与【{query}】相关的参考依据（知识库为空或无有效页面）。"}, ensure_ascii=False)
    query_parts = [
        part.strip().lower()
        for part in re.split(r"[\s，,；;、。/|]+", query)
        if part.strip()
    ]
    meaningful_parts = [part for part in query_parts if part not in _QUERY_STOPWORDS]
    if not meaningful_parts:
        return json.dumps({
            "evidence": [],
            "message": "查询词过于宽泛，请提供具体法律概念或规则名称。",
        }, ensure_ascii=False)

    keywords = set(meaningful_parts)
    term_ngrams: Dict[str, set[str]] = {}
    for part in meaningful_parts:
        if re.search(r"[\u3400-\u9fff]", part) and len(part) > 2:
            term_ngrams[part] = {
                part[index:index + 2] for index in range(len(part) - 1)
            }
            keywords.update(term_ngrams[part])
    candidates = []
    for page in pages:
        if filter_tag and page.get("tag") != filter_tag:
            continue
        text = re.sub(r"\s+", "", page["text"].lower())
        compact_query_parts = [re.sub(r"\s+", "", part) for part in meaningful_parts]
        exact_score = sum(
            text.count(keyword) * (len(keyword) + 1)
            for keyword in compact_query_parts if keyword
        )
        matched_ngrams = sorted([
            keyword for keyword in keywords.difference(meaningful_parts)
            if text.count(keyword)
        ], key=lambda keyword: (-len(keyword), keyword))
        score = exact_score + sum(min(text.count(keyword), 3) for keyword in matched_ngrams)
        matched_set = set(matched_ngrams)
        matched_terms = sum(
            compact_part in text
            or bool(term_ngrams.get(part, set()).intersection(matched_set))
            for part, compact_part in zip(meaningful_parts, compact_query_parts)
        )
        coverage = matched_terms / len(meaningful_parts)
        minimum_coverage = min(0.5, 2 / len(meaningful_parts))
        if coverage >= minimum_coverage and (exact_score or len(matched_ngrams) >= 2):
            candidates.append((score, coverage, page, matched_ngrams))
    if not candidates:
        tag_desc = f"[{filter_tag}类]" if filter_tag else ""
        return json.dumps({"evidence": [], "message": f"在权威参考文档 {tag_desc} 中未检索到与【{query}】相关的法条或合规规定。"}, ensure_ascii=False)

    results = []
    ordered = sorted(
        candidates,
        key=lambda item: (item[0], item[1], item[2]["doc_name"], -item[2]["page_num"]),
        reverse=True,
    )[:max(0, top_k)]
    for index, (_, _, page, matched_ngrams) in enumerate(ordered, 1):
        text = page["text"]
        matched_terms = [term for term in meaningful_parts if term]
        matched_terms.extend(matched_ngrams)
        matches = []
        for term in matched_terms:
            if not term:
                continue
            pattern = r"\s*".join(re.escape(char) for char in term)
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                matches.append((match.start(), len(term)))
        position = max(matches, key=lambda item: item[1], default=(-1, 0))[0]
        if position >= 0:
            start = max(0, position - 50)
            end = min(len(text), start + max_snippet_len)
            snippet = text[start:end]
            if start:
                snippet = "..." + snippet
            if end < len(text):
                snippet += "..."
        else:
            snippet = text[:max_snippet_len] + "..."
        results.append({
            "id": page.get("id", f"EV{index}"), "doc_name": page["doc_name"],
            "tag": page.get("tag"),
            "source_type": {
                "法": "law",
                "合规": "enterprise_document",
                "通用": "general_document",
            }.get(page.get("tag"), "reference_document"),
            "article_no": page.get("article_no"), "title": page.get("title", ""),
            "page_start": page.get("page_start", page.get("page_num")),
            "page_end": page.get("page_end", page.get("page_num")),
            "snippet": snippet, "text": text,
        })
    return json.dumps({"evidence": results}, ensure_ascii=False)
