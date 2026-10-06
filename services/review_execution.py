"""Shared execution semantics for single and concurrent review units."""

import json
from typing import Any, Callable, Dict, Iterable, Optional

from services.report_parser import parse_structured_report
from web.sse_client import stream_contract_review


ReviewEventHandler = Callable[[str, Dict[str, Any]], None]


def collect_evidence_observation(
    observation: str, destination: Dict[str, Dict[str, Any]]
) -> None:
    """Merge evidence records returned by a read-only search tool."""
    try:
        payload = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return
    for evidence in payload.get("evidence", []):
        if evidence.get("id"):
            evidence.setdefault("source_type", {
                "法": "law",
                "合规": "enterprise_document",
                "通用": "general_document",
            }.get(evidence.get("tag"), "law"))
            destination[evidence["id"]] = evidence


def execute_review_unit(
    base_url: str,
    contract_text: str,
    max_turns: int,
    review_side: str,
    on_event: Optional[ReviewEventHandler] = None,
    stream_factory: Callable[..., Iterable[Dict[str, Any]]] = stream_contract_review,
) -> Dict[str, Any]:
    """Consume one review stream and return its canonical result for either UI mode."""
    raw_report = ""
    final_complete = False
    evidence_records: Dict[str, Dict[str, Any]] = {}
    error = ""

    try:
        events = stream_factory(
            base_url,
            contract_text,
            max_turns,
            review_side=review_side,
        )
        for item in events:
            event = item.get("event", "")
            data = item.get("data", {})
            if on_event:
                on_event(event, data)
            if event == "tool_result":
                collect_evidence_observation(
                    data.get("observation", ""), evidence_records
                )
            elif event == "final_report":
                raw_report = data.get("raw_report", "")
                final_complete = bool(data.get("is_complete"))
                if not final_complete:
                    error = "未收到完整的最终审查报告"
            elif event == "error":
                error = data.get("error", "网关返回错误")
                break

        if not error and (not raw_report or not final_complete):
            error = "未收到完整的最终审查报告"
    except Exception as exc:
        error = str(exc)

    report = raw_report if raw_report and final_complete else ""
    if report:
        parsed_report = parse_structured_report(report)
        if not parsed_report or not parsed_report.reviews:
            report = ""
            error = "模型最终报告没有可用的结构化审查条目"
    success = bool(report) and not error
    if not success and not error:
        error = "最终报告为空"

    return {
        "success": success,
        "report": report,
        "evidence_records": evidence_records,
        "error": error,
    }


def aggregate_review_results(
    results: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep valid reports while returning failures for separate UI reporting."""
    reviews = []
    failures = []
    for result in results:
        title = result.get("title") or f"条款 {result.get('index', '')}".strip()
        if not result.get("success"):
            failures.append(f"{title}：{result.get('error') or '审查失败'}")
            continue

        parsed_report = parse_structured_report(result.get("report", ""))
        if not parsed_report or not parsed_report.reviews:
            failures.append(f"{title}：成功任务的报告无法解析为结构化条目")
            continue
        reviews.extend(
            item.model_dump(mode="json") for item in parsed_report.reviews
        )

    report = (
        json.dumps({"reviews": reviews}, ensure_ascii=False)
        if reviews
        else ""
    )
    return {"report": report, "failures": failures}
