"""Validate and finalize reports after the ReAct tool loop."""

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


class ReportFinalizer:
    """Repair report structure and add only verified enterprise citations."""

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
    def _decode_supplements(response: str) -> Dict[str, Dict[str, Any]]:
        start = response.find("{")
        end = response.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            payload = json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            return {}
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
    def _normalize_clause_topic(topic: str) -> str:
        return re.sub(
            r"^\s*(?:\d+(?:\.\d+)*(?:[.、)]?\s*)|第[一二三四五六七八九十百千万零〇两\d]+条\s*)",
            "",
            topic,
        ).strip()

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
            if any(
                record.get("source_type") == "enterprise_rule"
                and str(record.get("enterprise_risk_level", "")).strip().casefold()
                in {"high", "高", "高风险"}
                for _, record in records
            ):
                values["risk_level"] = "高风险"
            if values:
                supplements[topic] = values
        return supplements

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
        final_text = report
        evidence = tool_call_records.evidence_records()
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "format_validation",
            "status": "started",
            "message": "正在校验最终报告结构",
        })
        format_issues = validate_report_structure(final_text)
        repair_instructions = list(format_issues)
        original_placeholder_count = final_text.count("初稿未提供")
        if original_placeholder_count:
            repair_instructions.append(
                "尝试从同一条款初稿的其他段落归位已明确的信息，减少普通字段中的‘初稿未提供’；不得推断"
            )
        format_status = "completed"
        format_message = "最终报告结构校验通过"
        try:
            repaired = await self._generate_text(
                REPORT_FORMAT_REPAIR_PROMPT,
                json.dumps({"format_issues": repair_instructions, "report": final_text}, ensure_ascii=False),
                AGENT_CONFIG.get(
                    "report_format_max_tokens",
                    min(4096, max(1024, AGENT_CONFIG.get("max_context_tokens", 8192) // 2)),
                ),
            )
            repaired_issues = validate_report_structure(repaired) if repaired else format_issues
            improved = (
                len(repaired_issues) < len(format_issues)
                or (
                    repaired.count("初稿未提供") < original_placeholder_count
                    and len(repaired_issues) <= len(format_issues)
                )
            )
            if repaired and (not repaired_issues or improved):
                final_text = repaired
                format_message = (
                    "模型已校验最终报告格式"
                    if not format_issues
                    else "格式问题已由模型修复并复核"
                )
            elif format_issues:
                format_status = "failed"
                format_message = "模型未能修复全部格式问题，保留原报告并继续"
            else:
                format_message = "模型校验未改善已合格报告，保留原报告"
        except Exception as exc:
            format_status = "failed"
            format_message = f"格式复核模型调用失败，保留原报告并继续：{exc}"
        final_text = normalize_report_structure(final_text) or final_text
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "format_validation",
            "status": format_status,
            "message": format_message,
            "count": len(validate_report_structure(final_text)),
        })

        enterprise_evidence = {
            evidence_id: item
            for evidence_id, item in evidence.items()
            if item.get("source_type") in {"enterprise_document", "enterprise_rule"}
        }
        parsed_report = parse_structured_report(final_text)
        clauses_to_reconcile = [
            item.clause_topic
            for item in (parsed_report.reviews if parsed_report else [])
        ] if enterprise_evidence else []
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "report_reconciliation",
            "status": "started",
            "message": "正在依据工具调用记录复核风险等级和企业知识库依据",
            "count": len(tool_call_records.snapshot()["calls"]),
        })
        applied = {}
        supplement_status = "skipped"
        supplement_message = "没有待补充条款或企业知识库命中"
        if enterprise_evidence and clauses_to_reconcile:
            evidence_payload = [
                {
                    "id": evidence_id,
                    "source_type": item.get("source_type"),
                    "source_location": item.get("source_location") or item.get("source_path") or item.get("doc_name"),
                    "title": item.get("title"),
                    "enterprise_risk_level": item.get("enterprise_risk_level"),
                    "text": str(item.get("text") or "")[:1600],
                }
                for evidence_id, item in list(enterprise_evidence.items())[:20]
            ]
            try:
                response = await self._generate_text(
                    KNOWLEDGE_EVIDENCE_SUPPLEMENT_PROMPT,
                    json.dumps({
                        "contract_text": contract_text,
                        "report": final_text,
                        "enterprise_evidence": evidence_payload,
                        "tool_call_records": [
                            {
                                "turn": call["turn"],
                                "tool": call["tool"],
                                "arguments": call["arguments"],
                                "success": call["success"],
                                "evidence_ids": [
                                    item["id"] for item in call["evidence"]
                                ],
                            }
                            for call in tool_call_records.snapshot()["calls"]
                            if call["tool"] != "submit_final_report"
                        ],
                    }, ensure_ascii=False),
                    AGENT_CONFIG.get("knowledge_supplement_max_tokens", 512),
                )
                applied = self._validated_supplements(
                    final_text,
                    self._decode_supplements(response),
                    enterprise_evidence,
                )
                if applied:
                    final_text = normalize_report_structure(
                        final_text, evidence_supplements=applied
                    ) or final_text
                    supplement_message = "已补充并核验相关企业知识库来源"
                else:
                    supplement_message = "本次命中资料未能匹配到需要补充的条款"
                supplement_status = "completed"
            except Exception as exc:
                supplement_status = "failed"
                supplement_message = f"知识库补充失败，保留格式校验后的报告：{exc}"
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "report_reconciliation",
            "status": supplement_status,
            "message": supplement_message,
            "count": len(applied),
        })
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "review_complete",
            "status": "completed",
            "message": "审查收尾流水线完成",
        })
        tool_call_records.finish("success")
        yield encode_event("pipeline_stage", {
            "task_id": task_id,
            "stage": "tool_recording",
            "status": "completed",
            "message": "工具调用记录已用于报告复核",
            "count": len(tool_call_records.snapshot()["calls"]),
        })
        yield encode_event("final_report", {
            "task_id": task_id,
            "raw_report": final_text,
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
