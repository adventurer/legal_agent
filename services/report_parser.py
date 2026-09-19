"""模型审查报告解析与清洗工具。"""

import json
import re
from typing import Any, Optional, Tuple

from core.schemas import ContractReviewReport


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


def is_final_report(text: str) -> bool:
    """判断模型输出是否已经进入最终审查报告阶段。"""
    if not text:
        return False
    if any(marker in text for marker in ("Final:", "【最终结论】", "最终审查意见", "综合审查报告")):
        return True
    return sum(signature in text for signature in REPORT_SIGNATURES) >= 2


def clean_report_content(raw_text: str) -> str:
    """移除最终报告标记和正文中的代码围栏，保留围栏内文本内容。"""
    cleaned = (raw_text or "").strip()
    for prefix in ("Final:", "【最终结论】:", "【最终结论】", "最终审查意见:"):
        if prefix in cleaned:
            cleaned = cleaned.split(prefix, 1)[1].strip()
            break
    # 模型可能把合同原文嵌套在 ```markdown ... ``` 中。只移除围栏行，
    # 不删除围栏内的合同内容，避免 UI 将 Markdown 控制标记直接展示给用户。
    cleaned = re.sub(
        r"(?im)^[ \t]*```[ \t]*(?:markdown|md|text)?[ \t]*\r?$",
        "",
        cleaned,
    )
    return cleaned.strip()


def parse_structured_report(raw_text: str) -> Optional[ContractReviewReport]:
    """尽力将 JSON Final 结果解析为结构化报告；Markdown 结果返回 None。"""
    content = clean_report_content(raw_text)
    if not content:
        return None
    try:
        return ContractReviewReport.model_validate(json.loads(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def split_final_output(raw_text: str) -> Tuple[bool, str, Optional[ContractReviewReport]]:
    """返回是否为最终报告、清洗后的正文和可选的结构化报告。"""
    final = is_final_report(raw_text)
    content = clean_report_content(raw_text) if final else (raw_text or "").strip()
    return final, content, parse_structured_report(content) if final else None
