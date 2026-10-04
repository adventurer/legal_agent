"""Validate structured final-report submissions and deterministic acknowledgments."""

from typing import Iterable, Optional

from pydantic import ValidationError

from services.report_parser import missing_review_topics

from .contracts import FinalReportArguments, ModelToolCall


class FinalReportValidationError(ValueError):
    pass


def validate_final_report(
    call: ModelToolCall,
    required_guardrails: Iterable[str],
    finish_reason: Optional[str] = None,
    contract_text: str = "",
) -> FinalReportArguments:
    if finish_reason == "length":
        raise FinalReportValidationError(
            "最终报告触发输出长度上限，必须补全后重新提交。"
        )
    try:
        submission = FinalReportArguments.model_validate_json(call.arguments)
    except ValidationError as exc:
        raise FinalReportValidationError(str(exc)[:1500]) from exc

    missing_topics = missing_review_topics(submission.report, contract_text)
    if missing_topics:
        details = "、".join(missing_topics[:20])
        if len(missing_topics) > 20:
            details += f"等共 {len(missing_topics)} 项"
        raise FinalReportValidationError(
            f"最终报告遗漏合同审查条目：{details}"
        )

    required = set(required_guardrails)
    acknowledged = set(submission.acknowledged_guardrails)
    missing = sorted(required - acknowledged)
    unknown = sorted(acknowledged - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"遗漏必须核实的风险标记: {', '.join(missing)}")
        if unknown:
            details.append(f"包含未知风险标记: {', '.join(unknown)}")
        raise FinalReportValidationError("；".join(details))

    return submission