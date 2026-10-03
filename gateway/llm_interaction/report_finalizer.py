"""Finalize reports after the ReAct tool loop."""

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict

from starlette.concurrency import iterate_in_threadpool

from configs.config import AGENT_CONFIG
from core.prompts import (
    KNOWLEDGE_EVIDENCE_SUPPLEMENT_PROMPT,
    REPORT_FORMAT_REPAIR_PROMPT,
)
from services.report_parser import (
    normalize_report_structure,
    parse_structured_report,
    validate_report_structure,
)

from .contracts import FinalReportArguments
from .events import encode_event
from .tool_call_recorder import ToolCallRecorder


_ENTERPRISE_RISK_FLOORS = {
    "high": "高风险",
    "高": "高风险",
    "高风险": "高风险",
    "medium": "中风险",
    "med": "中风险",
    "中": "中风险",
    "中风险": "中风险",
    "low": "低风险",
    "低": "低风险",
    "低风险": "低风险",
    "notice": "提示",
    "提示": "提示",
}
_REPORT_RISK_RANK = {"提示": 0, "低风险": 1, "中风险": 2, "高风险": 3}


class ReportFinalizer:
    """Pass through model reports without post-processing."""

    def __init__(self, agent: Any):
        self.agent = agent

    async def _generate_text(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
    ) -> str:
        def create_response():
            try:
                return self.agent.client.chat.completions.create(
                    model=self.agent.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    stream=True,
                )
            except StopIteration as exc:
                raise RuntimeError("模型未返回格式校验响应") from exc

        response = await asyncio.to_thread(create_response)
        content_parts = []
        try:
            async for chunk in iterate_in_threadpool(response):
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    content_parts.append(content)
        finally:
            close = getattr(response, "close", None)
            if close:
                try:
                    await asyncio.to_thread(close)
                except Exception:
                    pass
        return "".join(content_parts).strip()

    @staticmethod
    def _decode_payload(response: str) -> Dict[str, Any]:
        start = response.find("{")
        end = response.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            payload = json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _decode_supplements(response: str) -> Dict[str, Dict[str, Any]]:
        payload = ReportFinalizer._decode_payload(response)
        supplements = payload.get("supplements", []) if isinstance(payload, dict) else []
        if not isinstance(supplements, list):
            return {}
        return {
            item["clause_topic"]: {
                "evidence_ids": item.get("evidence_ids", []),
                "raise_to_high_risk": item.get("raise_to_high_risk") is True,
            }
            for item in supplements
            if isinstance(item, dict) and isinstance(item.get("clause_topic"), str)
        }

    @staticmethod
    def _validated_rule_assessments(
        report: str,
        contract_text: str,
        assessments: Any,
        evidence: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
        expected_ids = {
            evidence_id
            for evidence_id, item in evidence.items()
            if item.get("source_type") == "enterprise_rule"
        }
        if not expected_ids:
            return {}, []
        parsed = parse_structured_report(report)
        if not parsed:
            return {}, ["最终报告无法解析，不能核验企业规则适用性"]
        if not isinstance(assessments, list):
            return {}, ["缺少 rule_assessments 数组"]

        reviews = {item.clause_topic: item for item in parsed.reviews}
        normalized_reviews: Dict[str, list[str]] = {}
        for topic in reviews:
            normalized_reviews.setdefault(
                ReportFinalizer._normalize_clause_topic(topic), []
            ).append(topic)

        proposals: Dict[str, Dict[str, Any]] = {}
        issues = []
        seen_ids = set()

        def grounded_contract_quote(value: Any) -> bool:
            quote = re.sub(r"\s+", "", str(value or ""))
            source = re.sub(r"\s+", "", contract_text)
            return bool(quote) and quote in source

        for assessment in assessments:
            if not isinstance(assessment, dict):
                issues.append("存在格式无效的企业规则核对记录")
                continue
            evidence_id = assessment.get("evidence_id")
            if evidence_id not in expected_ids:
                issues.append(f"核对记录包含未知规则编号：{evidence_id}")
                continue
            if evidence_id in seen_ids:
                issues.append(f"规则 {evidence_id} 被重复核对")
                continue
            seen_ids.add(evidence_id)

            applicability = assessment.get("applicability")
            clauses = assessment.get("applicable_clauses")
            if not isinstance(clauses, list):
                issues.append(f"规则 {evidence_id} 缺少 applicable_clauses 数组")
                continue
            if applicability == "not_applicable":
                if clauses or not str(assessment.get("not_applicable_reason", "")).strip():
                    issues.append(f"规则 {evidence_id} 的不适用结论缺少理由或与适用条款冲突")
                if not grounded_contract_quote(assessment.get("contract_basis")):
                    issues.append(f"规则 {evidence_id} 的不适用判断没有可在合同原文中核实的引文")
                continue
            if applicability != "applicable" or not clauses:
                issues.append(f"规则 {evidence_id} 未给出有效的适用结论")
                continue

            for clause in clauses:
                if not isinstance(clause, dict):
                    issues.append(f"规则 {evidence_id} 的适用条款记录格式无效")
                    continue
                topic = clause.get("clause_topic")
                review = reviews.get(topic)
                if review is None:
                    matches = normalized_reviews.get(
                        ReportFinalizer._normalize_clause_topic(str(topic or "")), []
                    )
                    if len(matches) == 1:
                        topic = matches[0]
                        review = reviews[topic]
                if review is None:
                    issues.append(f"规则 {evidence_id} 指向了不存在或不唯一的条款：{topic}")
                    continue
                required_values = (
                    clause.get("contract_basis"),
                    clause.get("reason"),
                    clause.get("recommendation_for_report"),
                )
                if not all(str(value or "").strip() for value in required_values):
                    issues.append(f"规则 {evidence_id} 的适用判断或建议不完整")
                    continue
                if not grounded_contract_quote(clause.get("contract_basis")):
                    issues.append(f"规则 {evidence_id} 的合同依据不在合同原文中")
                    continue
                proposal = proposals.setdefault(topic, {
                    "evidence_ids": [],
                    "recommendation_adoptions": [],
                    "rule_impacts": [],
                })
                proposal["evidence_ids"].append(evidence_id)
                proposal["recommendation_adoptions"].append(
                    str(clause["recommendation_for_report"]).strip()
                )
                proposal["rule_impacts"].append({
                    "evidence_id": evidence_id,
                    "enterprise_risk_level": evidence[evidence_id].get(
                        "enterprise_risk_level", ""
                    ),
                    "contract_basis": str(clause["contract_basis"]).strip(),
                    "reason": str(clause["reason"]).strip(),
                })

        missing_ids = expected_ids - seen_ids
        issues.extend(f"规则 {evidence_id} 未核对" for evidence_id in sorted(missing_ids))
        return proposals, issues

    @staticmethod
    def _normalize_clause_topic(topic: str) -> str:
        return re.sub(
            r"^\s*(?:\d+(?:\.\d+)*(?:[.、)]?\s*)|第[一二三四五六七八九十百千万零〇两\d]+条\s*)",
            "",
            topic,
        ).strip()

    @staticmethod
    def _without_report_risk_levels(report: str) -> str:
        return "\n".join(
            line for line in report.splitlines()
            if "**风险等级**" not in line
            and "**企业内部风险等级" not in line
        )

    @staticmethod
    def _remove_rule_citations_from_legal_basis(report: str) -> str:
        pattern = re.compile(
            r"(?im)^(?P<prefix>\s*[-*+]\s*(?:\*\*)?法律(?:/合规)?依据"
            r"(?:\*\*)?\s*[:：]\s*)(?P<value>.*)$"
        )

        def clean(match: re.Match[str]) -> str:
            value = re.sub(r"\[\[RULE:[^\]]+\]\]", "", match.group("value"))
            value = re.sub(r"^[\s；;，,]+|[\s；;，,]+$", "", value)
            return match.group("prefix") + (value or "未检索到直接依据")

        return pattern.sub(clean, report)

    @staticmethod
    def _validated_supplements(
        report: str,
        proposed: Dict[str, Dict[str, Any]],
        evidence: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        parsed = parse_structured_report(report)
        if not parsed:
            return {}
        valid_reviews = {item.clause_topic: item for item in parsed.reviews}
        normalized_reviews: Dict[str, list[Any]] = {}
        for topic, review in valid_reviews.items():
            normalized_reviews.setdefault(
                ReportFinalizer._normalize_clause_topic(topic), []
            ).append(review)
        missing_values = {
            "", "初稿未提供", "未检索到企业内部风险等级",
            "未检索到相关企业规则或知识库依据",
        }
        supplements = {}
        for topic, suggestion in proposed.items():
            review = valid_reviews.get(topic)
            if review is None:
                matches = normalized_reviews.get(
                    ReportFinalizer._normalize_clause_topic(topic), []
                )
                if len(matches) == 1:
                    review = matches[0]
            ids = suggestion.get("evidence_ids", [])
            if not review or not isinstance(ids, list):
                continue
            records = []
            for evidence_id in dict.fromkeys(ids):
                record = evidence.get(evidence_id)
                if isinstance(record, dict) and record.get("source_type") in {
                    "enterprise_document", "enterprise_rule",
                }:
                    records.append((evidence_id, record))
            values: Dict[str, str] = {}
            if records:
                references = []
                for evidence_id, record in records:
                    ref_type = "RULE" if record.get("source_type") == "enterprise_rule" else "KB"
                    location = (
                        record.get("source_location")
                        or record.get("source_path")
                        or record.get("doc_name")
                        or "企业知识库"
                    )
                    references.append(f"{location} [[{ref_type}:{evidence_id}]]")
                values["enterprise_basis"] = "；".join(references)
            grades = list(dict.fromkeys(
                str(record["enterprise_risk_level"])
                for _, record in records
                if record.get("source_type") == "enterprise_rule"
                and record.get("enterprise_risk_level")
            ))
            if grades:
                values["enterprise_risk_level"] = "、".join(grades)
            recommendations = [
                str(item).strip()
                for item in suggestion.get("recommendation_adoptions", [])
                if str(item).strip()
            ]
            if recommendations:
                existing_recommendation = str(review.suggested_revision or "").strip()
                values["suggested_revision"] = "\n".join(dict.fromkeys(
                    ([existing_recommendation] if existing_recommendation else [])
                    + [f"企业规则建议：{item}" for item in recommendations]
                ))
            rule_impacts = suggestion.get("rule_impacts", [])
            applicable_floors = [
                floor
                for _, record in records
                if record.get("source_type") == "enterprise_rule"
                and (floor := _ENTERPRISE_RISK_FLOORS.get(
                    str(record.get("enterprise_risk_level", "")).strip().casefold()
                ))
            ]
            if isinstance(rule_impacts, list) and rule_impacts:
                impact_notes = []
                for impact in rule_impacts:
                    if not isinstance(impact, dict):
                        continue
                    evidence_id = impact.get("evidence_id")
                    record = evidence.get(evidence_id, {})
                    raw_grade = str(
                        record.get("enterprise_risk_level")
                        or impact.get("enterprise_risk_level")
                        or ""
                    ).strip()
                    impact_notes.append(
                        f"{evidence_id}（企业内部等级：{raw_grade or '未提供'}）："
                        f"{impact.get('contract_basis', '')}；{impact.get('reason', '')}"
                    )
                if impact_notes:
                    existing_issue = str(review.issue or "").strip()
                    values["issue"] = "\n".join(dict.fromkeys(
                        ([existing_issue] if existing_issue else [])
                        + ["企业规则核对：" + note for note in impact_notes]
                    ))
            current_risk = getattr(review.risk_level, "value", review.risk_level)
            current_risk = _ENTERPRISE_RISK_FLOORS.get(
                str(current_risk).strip().casefold(),
                str(current_risk).strip(),
            )
            if applicable_floors:
                floor = max(
                    applicable_floors,
                    key=lambda risk: _REPORT_RISK_RANK[risk],
                )
                current_rank = _REPORT_RISK_RANK.get(current_risk, -1)
                if _REPORT_RISK_RANK[floor] > current_rank:
                    values["risk_level"] = floor
            if values:
                supplements[topic] = values
        return supplements

    @staticmethod
    def _enforce_enterprise_risk_levels(
        report: str,
        evidence: Dict[str, Dict[str, Any]],
    ) -> str:
        parsed = parse_structured_report(report)
        if not parsed:
            return report

        changed = False
        for review in parsed.reviews:
            cited_rule_ids = re.findall(
                r"\[\[RULE:(RULE\d+)\]\]",
                review.enterprise_basis or "",
            )
            grades = list(dict.fromkeys(
                str(record["enterprise_risk_level"])
                for rule_id in cited_rule_ids
                if (record := evidence.get(rule_id))
                and record.get("source_type") == "enterprise_rule"
                and record.get("enterprise_risk_level")
            ))
            if grades:
                authoritative_grade = "、".join(grades)
                if review.enterprise_risk_level != authoritative_grade:
                    review.enterprise_risk_level = authoritative_grade
                    changed = True
        return parsed.model_dump_json(ensure_ascii=False) if changed else report

    async def finalize(
        self,
        report: str,
        contract_text: str,
        task_id: str,
        turn: int,
        acknowledged_guardrails: list[str],
        finish_reason: str | None,
        tool_call_records: ToolCallRecorder,
    ) -> AsyncGenerator[Dict[str, str], None]:
        report = self._enforce_enterprise_risk_levels(
            report,
            tool_call_records.evidence_records(),
        )
        call_count = len(tool_call_records.snapshot()["calls"])
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "format_validation",
            "status": "skipped",
            "message": "已关闭报告格式校验，直接采用大模型原始输出",
            "count": 0,
        })
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "report_reconciliation",
            "status": "skipped",
            "message": "已关闭企业规则核对，直接输出模型提交报告",
            "count": call_count,
        })
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "review_complete",
            "status": "completed",
            "message": "已接收模型提交报告",
        })
        tool_call_records.finish("success")
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "tool_recording",
            "status": "completed",
            "message": "工具调用记录已保留",
            "count": call_count,
        })
        yield encode_event("final_report", {
            "task_id": task_id,
            "raw_report": report,
            "turns": turn,
            "is_complete": True,
            "status": "success",
            "acknowledged_guardrails": acknowledged_guardrails,
            "finish_reason": finish_reason,
        })
        yield encode_event("done", {
            "task_id": task_id,
            "message": "审查流水线结束",
        })
