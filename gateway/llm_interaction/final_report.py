"""Validate structured final-report submissions and deterministic acknowledgments."""

import json
from typing import Iterable, Optional

from pydantic import ValidationError

from core.schemas import ContractReviewReport

from services.report_parser import missing_review_topics

from .contracts import FinalReportArguments, ModelToolCall


class FinalReportValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        issue_count: int = 0,
        issue_types: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.issue_count = issue_count
        self.issue_types = issue_types


def recover_final_report(
    call: ModelToolCall,
    required_guardrails: Iterable[str],
) -> Optional[tuple[FinalReportArguments, int, int]]:
    try:
        payload = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    report = payload.get("report")
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError:
            report_text = report.lstrip()
            try:
                decoded_report, end = json.JSONDecoder().raw_decode(report_text)
            except json.JSONDecodeError:
                return None
            if report_text[end:].strip() not in {"}", "]"}:
                return None
            report = decoded_report
    if not isinstance(report, dict) or not isinstance(report.get("reviews"), list):
        return None

    field_defaults = {
        "clause_topic": "未提供条款主题",
        "legal_basis": "未提供法律依据",
        "issue": "未提供风险说明",
        "suggested_revision": "未提供修改建议",
    }
    normalized_reviews = []
    normalized_field_count = 0
    dropped_review_count = 0
    for review in report["reviews"]:
        if not isinstance(review, dict):
            dropped_review_count += 1
            continue
        normalized = dict(review)
        for field, fallback in field_defaults.items():
            value = normalized.get(field)
            if not isinstance(value, str) or not value.strip():
                normalized[field] = fallback
                normalized_field_count += 1
        normalized_reviews.append(normalized)
    if not normalized_reviews:
        return None

    try:
        parsed_report = ContractReviewReport.model_validate(
            {"reviews": normalized_reviews}
        )
        allowed_guardrails = set(required_guardrails)
        acknowledged = payload.get("acknowledged_guardrails", [])
        if not isinstance(acknowledged, list):
            acknowledged = []
        acknowledged = list(dict.fromkeys(
            code for code in acknowledged
            if isinstance(code, str) and code in allowed_guardrails
        ))
        submission = FinalReportArguments.model_validate({
            "report": parsed_report,
            "acknowledged_guardrails": acknowledged,
        })
    except ValidationError:
        return None
    return submission, normalized_field_count, dropped_review_count


def validate_final_report(
    call: ModelToolCall,
    required_guardrails: Iterable[str],
    finish_reason: Optional[str] = None,
    contract_text: str = "",
) -> FinalReportArguments:
    if finish_reason == "length":
        raise FinalReportValidationError(
            "最终报告触发输出长度上限，必须补全后重新提交。",
            reason_code="generation_length_limit",
        )
    try:
        submission = FinalReportArguments.model_validate_json(call.arguments)
    except ValidationError as original_error:
        validation_error: Optional[ValidationError] = original_error
        errors = original_error.errors()
        if (
            len(errors) == 1
            and errors[0].get("type") == "model_type"
            and errors[0].get("loc") == ("report",)
        ):
            try:
                payload = json.loads(call.arguments)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("report"), str):
                try:
                    decoded_report = json.loads(payload["report"])
                except json.JSONDecodeError as report_error:
                    report_text = payload["report"].lstrip()
                    try:
                        decoded_report, end = json.JSONDecoder().raw_decode(
                            report_text
                        )
                    except json.JSONDecodeError:
                        decoded_report = None
                        trailing_text = ""
                    else:
                        trailing_text = report_text[end:].strip()
                    if not isinstance(decoded_report, dict) or trailing_text != "}":
                        raise FinalReportValidationError(
                            "report 字段是字符串且其内容不是有效 JSON 对象；请提交 report 对象。",
                            reason_code="report_string_unrecoverable",
                            issue_count=1,
                            issue_types=("json_invalid",),
                        ) from report_error
                payload["report"] = decoded_report
                normalized_arguments = json.dumps(payload, ensure_ascii=False)
                try:
                    submission = FinalReportArguments.model_validate_json(
                        normalized_arguments
                    )
                except ValidationError as normalized_error:
                    validation_error = normalized_error
                else:
                    validation_error = None

        if validation_error is not None:
            errors = validation_error.errors()
            issue_types = tuple(sorted({
                str(error.get("type", "unknown"))
                for error in errors
            }))
            raise FinalReportValidationError(
                str(validation_error)[:1500],
                reason_code="schema_validation",
                issue_count=len(errors),
                issue_types=issue_types,
            ) from validation_error

    missing_topics = missing_review_topics(submission.report, contract_text)
    if missing_topics:
        details = "、".join(missing_topics[:20])
        if len(missing_topics) > 20:
            details += f"等共 {len(missing_topics)} 项"
        raise FinalReportValidationError(
            f"最终报告遗漏合同审查条目：{details}",
            reason_code="missing_review_topics",
            issue_count=len(missing_topics),
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
        raise FinalReportValidationError(
            "；".join(details),
            reason_code="guardrail_acknowledgment_mismatch",
            issue_count=len(missing) + len(unknown),
        )

    return submission