"""Search scoring and presentation for enterprise review rules."""

from typing import Any, Iterable, List


def search_rules(rows: Iterable[Any], query: str, top_k: int = 2) -> str:
    if not query or not query.strip():
        return "未提供有效的查询关键词。"
    tokens = [
        token.strip()
        for token in query.replace(",", " ").replace("，", " ").split()
        if token.strip()
    ] or [query.strip()]
    rows = list(rows)
    if not rows:
        return "企业自编法典库当前为空。"

    scored = []
    for row in rows:
        score = 0
        for token in tokens:
            if token in row["topic"]:
                score += 5
            if token in row["keywords"]:
                score += 3
            if token in row["standard_requirement"]:
                score += 1
            if token in (row["forbidden_pattern"] or ""):
                score += 1
        if score:
            scored.append((score, row))
    if not scored:
        return f"《企业自编法典》中暂未收录针对【{query}】的特殊禁止性规则与审查偏好。"

    results: List[str] = []
    for index, (_, rule) in enumerate(sorted(scored, key=lambda item: item[0], reverse=True)[:top_k], 1):
        text = (
            f"【自编法典规则 {index}】主题：{rule['topic']} (控制红线级别: {rule['risk_level']})\n"
            f"- 企业控制要求: {rule['standard_requirement']}\n"
        )
        if rule["forbidden_pattern"]:
            text += f"- 严禁模式: {rule['forbidden_pattern']}\n"
        if rule["recommended_clause"]:
            text += f"- 推荐合规范本: {rule['recommended_clause']}"
        results.append(text.strip())
    return "\n\n".join(results)
