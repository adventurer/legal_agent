"""模型审查报告解析与清洗工具。"""

import json
import re
from typing import Dict, Optional, Tuple

from core.schemas import ContractReviewReport
from core.prompts import RISK_LEVEL_REPORT_LEGEND
from services.clause_splitter import ClauseSplitter


REPORT_SIGNATURES = (
    "风险等级",
    "修改建议",
    "法律依据",
    "合规依据",
    "审查结论",
    "### 1.",
    "1. 违约",
    "1. 争议",
)

REPORT_FIELD_PATTERNS = (
    ("风险类型", r"风险类型"),
    ("风险等级", r"风险等级"),
    ("企业内部风险等级", r"企业(?:内部)?风险等级"),
    ("法律效力", r"法律效力"),
    ("商业后果", r"商业后果"),
    ("救济成本", r"救济成本"),
    ("受影响方", r"受影响方|影响受方"),
    ("结论置信度", r"结论置信度|证据置信度"),
    ("法律/合规依据", r"法律/合规依据|法律依据|合规依据"),
    ("企业知识库依据", r"企业(?:内部|知识库)?依据|业务资料依据"),
    ("风险剖析", r"风险剖析|风险说明|风险分析"),
    ("修改建议", r"修改建议|建议修改"),
)

REPORT_SOURCE_FIELD_PATTERNS = {
    "risk_type": r"风险类型",
    "risk_level": r"风险等级",
    "enterprise_risk_level": r"企业(?:内部)?风险等级",
    "legal_effect": r"法律效力",
    "commercial_impact": r"商业后果",
    "remedy_cost": r"救济成本",
    "affected_party": r"受影响方|影响受方",
    "confidence": r"结论置信度|证据置信度",
    "legal_basis": r"法律(?:/合规)?依据|合规依据",
    "enterprise_basis": r"企业(?:内部|知识库)?依据|业务资料依据",
    "issue": r"风险剖析|风险说明|风险分析",
    "suggested_revision": r"修改建议|建议修改",
}

NUMBERED_HEADING_PATTERN = (
    r"^(?:\d+(?:\.\d+)*(?:[.、)]?)(?:\s|$)"
    r"|第[一二三四五六七八九十百千万零〇两\d]+条(?:\s|$))"
)
SOURCE_REVIEW_HEADING_PATTERN = re.compile(
    r"^\s*(?P<number>(?:\d+(?:\.\d+)*(?:[.、．)]?)|[（(]\d+[）)]|"
    r"[（(][一二三四五六七八九十百千万]+[）)]|"
    r"[一二三四五六七八九十百千万]+[、.．）)]))\s*(?P<title>\S.*)$"
)
SOURCE_ARTICLE_HEADING_PATTERN = re.compile(
    r"^\s*(?P<number>第[一二三四五六七八九十百千万零〇两\d]+条)"
    r"\s*(?P<title>\S.*)$"
)


def is_final_report(text: str) -> bool:
    """判断模型输出是否已经进入最终审查报告阶段。"""
    if not text:
        return False
    if any(marker in text for marker in (
        "Final:", "【最终结论】", "最终审查意见", "综合审查报告", "最终报告如下",
    )):
        return True
    if sum(signature in text for signature in REPORT_SIGNATURES) >= 2:
        return True
    try:
        payload = json.loads(clean_report_content(text))
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("reviews"), list)


def clean_report_content(raw_text: str) -> str:
    """移除最终报告标记和正文中的代码围栏，保留围栏内文本内容。"""
    cleaned = (raw_text or "").strip()
    for prefix in (
        "最终报告如下：", "最终报告如下:", "Final:",
        "【最终结论】:", "【最终结论】", "最终审查意见:",
    ):
        if prefix in cleaned:
            cleaned = cleaned.split(prefix, 1)[1].strip()
            break
    # 模型可能把合同原文嵌套在 ```markdown ... ``` 中。只移除围栏行，
    # 不删除围栏内的合同内容，避免 UI 将 Markdown 控制标记直接展示给用户。
    cleaned = re.sub(
        r"(?im)^[ \t]*```[ \t]*(?:json|markdown|md|text)?[ \t]*\r?$",
        "",
        cleaned,
    )
    return cleaned.strip()


def restore_clause_numbers(
    report: ContractReviewReport, contract_text: str
) -> ContractReviewReport:
    """Restore omitted source numbering when a report title has one unique match."""
    numbered_headings = []
    heading_pattern = re.compile(
        r"^(?P<number>第[一二三四五六七八九十百千万零〇两\d]+[条章]|"
        r"\d+(?:\.\d+)*(?:[.、)]?)|[（(]\d+[）)]|"
        r"[一二三四五六七八九十百]+、)\s*(?P<title>\S.*)$"
    )

    for clause in ClauseSplitter.split(contract_text):
        if clause.clause_type != "article":
            continue
        source_headings = [clause.title, *clause.content.splitlines()]
        for source_heading in source_headings:
            match = heading_pattern.match(source_heading.strip())
            if match:
                source_title = re.split(
                    r"[：:]", match.group("title"), maxsplit=1
                )[0].strip()
                numbered_headings.append(
                    (match.group("number"), source_title)
                )

    def normalize_title(title: str) -> str:
        return re.sub(r"[\s\W_]+", "", title, flags=re.UNICODE).casefold()

    titles_by_key: Dict[str, list[str]] = {}
    for number, title in numbered_headings:
        key = normalize_title(title)
        if key:
            titles_by_key.setdefault(key, []).append(number)

    for review in report.reviews:
        topic = review.clause_topic.strip()
        topic_heading = heading_pattern.match(topic)
        title = topic_heading.group("title") if topic_heading else topic
        matches = list(dict.fromkeys(
            titles_by_key.get(normalize_title(title), [])
        ))
        if len(matches) == 1:
            source_number = matches[0]
            if not topic_heading or topic_heading.group("number") != source_number:
                review.clause_topic = f"{source_number} {title}"
    return report


def missing_review_topics(
    report: ContractReviewReport, contract_text: str
) -> list[str]:
    """Return source headings not represented by distinct final review items."""
    expected = []
    for clause in ClauseSplitter.split(contract_text):
        if clause.clause_type != "article":
            continue

        child_headings = []
        for line in clause.content.splitlines():
            match = SOURCE_REVIEW_HEADING_PATTERN.match(line.strip())
            if match:
                title = re.split(r"[：:]", match.group("title"), maxsplit=1)[0].strip()
                child_headings.append((match.group("number"), title))
        if child_headings:
            decimal_headings = [
                item for item in child_headings
                if re.fullmatch(
                    r"\d+(?:\.\d+)+[.、．)]?",
                    item[0],
                )
            ]
            if decimal_headings:
                min_depth = min(item[0].rstrip(".、．)").count(".") for item in decimal_headings)
                expected.extend(
                    item for item in decimal_headings
                    if item[0].rstrip(".、．)").count(".") == min_depth
                )
                continue

            simple_arabic_headings = [
                item for item in child_headings
                if re.fullmatch(r"\d+[、.．)]?", item[0])
            ]
            if simple_arabic_headings:
                expected.extend(simple_arabic_headings)
                continue

            chinese_headings = [
                item for item in child_headings
                if re.fullmatch(
                    r"(?:[一二三四五六七八九十百千万]+[、.．）)]|"
                    r"[（(][一二三四五六七八九十百千万]+[）)])",
                    item[0],
                )
            ]
            if chinese_headings:
                expected.extend(chinese_headings)
                continue

            expected.extend(
                item for item in child_headings
                if re.fullmatch(r"[（(]\d+[）)]", item[0])
            )
            continue

        article_heading = SOURCE_ARTICLE_HEADING_PATTERN.match(clause.title.strip())
        if article_heading:
            expected.append((
                article_heading.group("number"),
                article_heading.group("title").strip(),
            ))

    if not expected:
        return [] if report.reviews or not contract_text.strip() else ["合同正文"]

    title_counts: Dict[str, int] = {}
    for _, title in expected:
        normalized = re.sub(r"[\s\W_]+", "", title, flags=re.UNICODE).casefold()
        if normalized:
            title_counts[normalized] = title_counts.get(normalized, 0) + 1

    used_review_indices = set()
    missing = []
    for number, title in expected:
        normalized_title = re.sub(
            r"[\s\W_]+", "", title, flags=re.UNICODE
        ).casefold()
        matching_index = next(
            (
                index
                for index, review in enumerate(report.reviews)
                if index not in used_review_indices
                and (
                    re.search(
                        rf"(?<![\w.]){re.escape(number)}(?![\w.])",
                        review.clause_topic,
                    )
                    or (
                        normalized_title
                        and title_counts.get(normalized_title) == 1
                        and normalized_title
                        in re.sub(
                            r"[\s\W_]+",
                            "",
                            review.clause_topic,
                            flags=re.UNICODE,
                        ).casefold()
                    )
                )
            ),
            None,
        )
        if matching_index is None:
            missing.append(f"{number} {title}".strip())
        else:
            used_review_indices.add(matching_index)
    return missing


def validate_report_structure(raw_text: str) -> list[str]:
    """Return format violations against the required Markdown report structure."""
    content = clean_report_content(raw_text)
    issues = []
    if not content.startswith(RISK_LEVEL_REPORT_LEGEND):
        issues.append("风险等级提示缺失、不完整或未置于报告开头")

    headings = list(re.finditer(r"(?m)^[ \t]*#{1,6}\s+(.+?)\s*$", content))
    placeholder_headings = [
        heading for heading in headings
        if re.search(r"\[.*(?:条款名称|主题).*\]|条款名称/主题", heading.group(1))
    ]
    if placeholder_headings:
        issues.append("报告中残留模板占位标题，须替换为合同原条款编号和名称")
    numbered_headings = [
        (index, heading)
        for index, heading in enumerate(headings)
        if heading not in placeholder_headings
        if re.match(
            r"^(?:\d+(?:\.\d+)*(?:[.、)]?)(?:\s|$)|第[一二三四五六七八九十百千万零〇两\d]+条(?:\s|$))",
            heading.group(1).strip(),
        )
    ]
    if not numbered_headings:
        issues.append("缺少按合同原编号排列的条款标题")
        return issues

    for position, (_, heading) in enumerate(numbered_headings, start=1):
        next_heading_index = (
            numbered_headings[position][0]
            if position < len(numbered_headings)
            else len(headings)
        )
        section_end = (
            headings[next_heading_index].start()
            if next_heading_index < len(headings)
            else len(content)
        )
        section = content[heading.end():section_end]
        field_positions = []
        missing_fields = []
        for label, pattern in REPORT_FIELD_PATTERNS:
            match = re.search(
                rf"(?im)^\s*(?:#{{1,6}}\s*)?[-*+]??\s*(?:\*\*)?(?:{pattern})(?:\*\*\s*[:：]|\s*[:：]\s*(?:\*\*)|\s*[:：])\s*[:：]?",
                section,
            )
            if match:
                field_positions.append((label, match.start()))
            else:
                missing_fields.append(label)
        topic = heading.group(1).strip()
        if missing_fields:
            issues.append(f"条款“{topic}”缺少字段：{'、'.join(missing_fields)}")
        if len(field_positions) == len(REPORT_FIELD_PATTERNS):
            actual_order = [label for label, _ in sorted(field_positions, key=lambda item: item[1])]
            expected_order = [label for label, _ in REPORT_FIELD_PATTERNS]
            if actual_order != expected_order:
                issues.append(f"条款“{topic}”字段顺序与模板不一致")
    return issues


def normalize_report_structure(
    raw_text: str,
    evidence_supplements: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[str]:
    """Reformat existing clause content without generating legal analysis."""
    def article_title(raw_title: str) -> Optional[str]:
        wrapped = re.match(
            r"^\d+(?:\.\d+)*(?:[.、)]?)\s*[\[【](第[一二三四五六七八九十百千万零〇两\d]+条.+?)[\]】]$",
            raw_title.strip(),
        )
        title = wrapped.group(1).strip() if wrapped else raw_title.strip(" []【】")
        if re.match(
            r"^第[一二三四五六七八九十百千万零〇两\d]+条(?:\s|$)", title
        ):
            return title
        return None

    content = clean_report_content(raw_text)
    headings = list(re.finditer(r"(?m)^[ \t]*#{1,6}\s+(.+?)\s*$", content))
    numbered = [
        (index, heading)
        for index, heading in enumerate(headings)
        if not re.search(r"\[.*(?:条款名称|主题).*\]|条款名称/主题", heading.group(1))
        if re.match(NUMBERED_HEADING_PATTERN, heading.group(1).strip())
    ]
    if not numbered:
        return None

    candidates = []
    article_context_by_heading = {}
    for position, (heading_index, heading) in enumerate(numbered):
        title = heading.group(1).strip()
        normalized_article_title = article_title(title)
        title = normalized_article_title or title.strip(" []【】")
        is_article_heading = normalized_article_title is not None
        next_article_position = next(
            (
                child_position
                for child_position in range(position + 1, len(numbered))
                if article_title(numbered[child_position][1].group(1))
            ),
            len(numbered),
        )
        child_headings = numbered[position + 1:next_article_position]
        has_numbered_child = is_article_heading and bool(child_headings)
        if has_numbered_child:
            first_child = child_headings[0][1]
            article_context_by_heading[headings.index(first_child)] = content[
                heading.end():first_child.start()
            ]
        if title and not has_numbered_child:
            candidates.append((heading_index, heading, title))
    if not candidates:
        return None

    required_fields = (
        ("风险类型", "risk_type"),
        ("风险等级", "risk_level"),
        ("企业内部风险等级", "enterprise_risk_level"),
        ("法律效力", "legal_effect"),
        ("商业后果", "commercial_impact"),
        ("救济成本", "remedy_cost"),
        ("受影响方", "affected_party"),
        ("结论置信度", "confidence"),
        ("法律/合规依据", "legal_basis"),
        ("企业知识库依据", "enterprise_basis"),
        ("风险剖析", "issue"),
        ("修改建议", "suggested_revision"),
    )
    missing_field_defaults = {
        "enterprise_risk_level": "未检索到企业内部风险等级",
        "enterprise_basis": "未检索到相关企业规则或知识库依据",
    }
    evidence_supplements = evidence_supplements or {}
    output_sections = []
    for position, (heading_index, heading, title) in enumerate(candidates):
        next_heading_index = (
            candidates[position + 1][0]
            if position + 1 < len(candidates)
            else len(headings)
        )
        section_end = (
            headings[next_heading_index].start()
            if next_heading_index < len(headings)
            else len(content)
        )
        section = (
            (content[:heading.start()] if position == 0 else "")
            + article_context_by_heading.get(heading_index, "")
            + content[heading.end():section_end]
        )
        lines = section.splitlines()
        markers = []
        for line_index, line in enumerate(lines):
            for field, pattern in REPORT_SOURCE_FIELD_PATTERNS.items():
                match = re.match(
                    rf"^\s*(?:#{{1,6}}\s*)?(?:[-*+]\s*)?(?:\*\*)?(?:{pattern})(?:\*\*\s*[:：]|\s*[:：]\s*(?:\*\*)|\s*[:：])\s*[:：]?\s*(.*)$",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    markers.append((line_index, field, match.group(1).strip()))
                    break

        values = {}
        covered_lines = set()
        for marker_index, (line_index, field, first_line) in enumerate(markers):
            end_line = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(lines)
            value_lines = ([first_line] if first_line else []) + [
                line.strip() for line in lines[line_index + 1:end_line] if line.strip()
            ]
            value = "\n".join(value_lines).strip()
            if value:
                values[field] = "\n".join(
                    part for part in (values.get(field), value) if part
                )
            covered_lines.update(range(line_index, end_line))

        unlabelled_content = "\n".join(
            line.strip() for index, line in enumerate(lines)
            if index not in covered_lines
            and line.strip()
            and not line.lstrip().startswith("#")
            and line.strip() not in {"---", "***"}
        ).strip()
        if unlabelled_content:
            values["issue"] = "\n".join(
                part for part in (values.get("issue"), unlabelled_content) if part
            )

        fields = [f"### {title}"]
        normalized_title = re.sub(r"^\s*\d+[.、)]\s*", "", title).strip()
        supplement = evidence_supplements.get(title) or evidence_supplements.get(
            normalized_title, {}
        )
        for label, field in required_fields:
            value = (
                supplement.get(field)
                if field in {
                    "risk_level", "enterprise_risk_level", "enterprise_basis",
                    "suggested_revision", "issue",
                } and supplement.get(field)
                else values.get(field)
            )
            if not value or value in {
                "初稿未提供",
                "未检索到企业内部风险等级",
                "未检索到相关企业规则或知识库依据",
            }:
                value = supplement.get(field) or missing_field_defaults.get(
                    field, "初稿未提供"
                )
            field_label = (
                f"**{label}：**"
                if field == "enterprise_risk_level"
                else f"**{label}**:"
            )
            fields.append(f"- {field_label} {value}")
        output_sections.append("\n".join(fields))

    return RISK_LEVEL_REPORT_LEGEND + "\n\n" + "\n\n".join(output_sections)


def parse_structured_report(raw_text: str) -> Optional[ContractReviewReport]:
    """解析 JSON 报告，或解析系统约定格式的 Markdown 条目。"""
    content = clean_report_content(raw_text)
    if not content:
        return None
    try:
        return ContractReviewReport.model_validate(json.loads(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    headings = list(re.finditer(r"(?m)^[ \t]*#{1,6}\s+(.+?)\s*$", content))
    reviews = []
    field_patterns = {
        "risk_level": r"风险等级",
        "enterprise_risk_level": r"企业(?:内部)?风险等级",
        "risk_type": r"风险类型",
        "legal_effect": r"法律效力",
        "commercial_impact": r"商业后果",
        "remedy_cost": r"救济成本",
        "affected_party": r"受影响方|影响受方",
        "confidence": r"结论置信度|证据置信度",
        "legal_basis": r"法律(?:/合规)?依据|合规依据",
        "enterprise_basis": r"企业(?:内部|知识库)?依据|业务资料依据",
        "issue": r"风险剖析|风险说明|风险分析",
        "suggested_revision": r"修改建议|建议修改",
    }
    required_fields = {"risk_level", "legal_basis", "issue", "suggested_revision"}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(content)
        section = content[heading.end():end]
        topic = re.sub(r"^\s*\d+[.、)]\s*", "", heading.group(1)).strip(" []【】")
        values = {}
        for field, label_pattern in field_patterns.items():
            match = re.search(
                rf"(?im)^\s*(?:#{{1,6}}\s*)?[-*+]?\s*(?:\*\*)?(?:{label_pattern})(?:\*\*\s*[:：]|\s*[:：]\s*(?:\*\*)|\s*[:：])\s*[:：]?\s*(.+?)\s*$",
                section,
            )
            if match:
                values[field] = match.group(1).strip().strip("* ")
        if topic and required_fields.issubset(values):
            reviews.append({"clause_topic": topic, **values})
    if not reviews:
        return None
    try:
        return ContractReviewReport.model_validate({"reviews": reviews})
    except (TypeError, ValueError):
        return None


def render_report_article(report: ContractReviewReport) -> str:
    """Render structured report data as a readable Markdown article."""
    risk_labels = {
        "High": "高风险",
        "Medium": "中风险",
        "Low": "低风险",
        "Notice": "履约/商务提示",
    }
    sections = ["# 合同审查报告", "", RISK_LEVEL_REPORT_LEGEND]
    if not report.reviews:
        sections.extend(["", "未生成条款审查条目。"])
        return "\n\n".join(sections)

    for index, item in enumerate(report.reviews, start=1):
        risk_level = getattr(item.risk_level, "value", item.risk_level)
        fields = (
            ("风险类型", item.risk_type),
            ("风险等级", risk_labels.get(str(risk_level), str(risk_level))),
            ("企业内部风险等级", item.enterprise_risk_level or "未检索到企业内部风险等级"),
            ("法律效力", item.legal_effect),
            ("商业后果", item.commercial_impact),
            ("救济成本", item.remedy_cost),
            ("受影响方", item.affected_party),
            ("结论置信度", item.confidence),
            ("法律/合规依据", item.legal_basis or "未检索到直接依据"),
            (
                "企业知识库依据",
                item.enterprise_basis or "未检索到相关企业规则或知识库依据",
            ),
            ("风险剖析", item.issue),
            ("修改建议", item.suggested_revision),
        )
        lines = [f"## {index}. {item.clause_topic}"]
        for label, value in fields:
            if value is None or not str(value).strip():
                continue
            formatted_value = str(value).strip().replace("\r\n", "\n").replace("\n", "\n  ")
            lines.append(f"- **{label}**: {formatted_value}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def split_final_output(raw_text: str) -> Tuple[bool, str, Optional[ContractReviewReport]]:
    """返回是否为最终报告、清洗后的正文和可选的结构化报告。"""
    final = is_final_report(raw_text)
    content = clean_report_content(raw_text) if final else (raw_text or "").strip()
    return final, content, parse_structured_report(content) if final else None
