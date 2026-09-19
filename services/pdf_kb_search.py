"""Keyword scoring and result formatting for PDF knowledge pages."""

from typing import Any, Dict, List, Optional


def search_pages(
    pages: List[Dict[str, Any]], query: str, filter_tag: Optional[str] = None,
    top_k: int = 2, max_snippet_len: int = 350,
) -> str:
    if not pages:
        return f"未找到任何与【{query}】相关的参考依据（知识库为空或无有效页面）。"
    keywords = [kw.strip() for kw in query.split() if kw.strip()] or [query.strip()]
    candidates = []
    for page in pages:
        if filter_tag and filter_tag not in page["tag"] and filter_tag not in page["doc_name"]:
            continue
        score = sum(page["text"].count(keyword) for keyword in keywords)
        if score:
            candidates.append((score, page))
    if not candidates:
        tag_desc = f"[{filter_tag}类]" if filter_tag else ""
        return f"在权威参考文档 {tag_desc} 中未检索到与【{query}】相关的法条或合规规定。"

    results = []
    for index, (_, page) in enumerate(sorted(candidates, key=lambda item: item[0], reverse=True)[:top_k], 1):
        text = page["text"]
        position = next((text.find(keyword) for keyword in keywords if text.find(keyword) != -1), -1)
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
        results.append(f"依据 [{index}] 《{page['doc_name']}》 第 {page['page_num']} 页:\n{snippet}")
    return "\n\n".join(results)
