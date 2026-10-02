"""Typed SSE event contracts shared by the API and review UI."""

from typing import Any, Dict, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from .contracts import GuardrailFinding


class EventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task_id: str


class StartPayload(EventPayload):
    message: str
    trace_id: Optional[str] = None


class TokenPayload(EventPayload):
    token: str


class ReportTokenPayload(EventPayload):
    token: str


class ToolStartPayload(EventPayload):
    tool: str
    call_id: str
    query: Optional[str] = None


class EvidenceSourcePayload(BaseModel):
    evidence_id: str
    source_type: str
    source_location: str


class ToolResultPayload(EventPayload):
    tool: str
    call_id: str
    observation: str
    success: bool
    elapsed_ms: int
    injected_chars: int = 0
    evidence_sources: list[EvidenceSourcePayload] = Field(default_factory=list)
    risk_evidence_ids: list[str] = Field(default_factory=list)
    enterprise_risk_levels: Dict[str, str] = Field(default_factory=dict)
    high_risk_evidence_ids: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class GuardrailPayload(EventPayload):
    findings: list[GuardrailFinding]


class ModelUsagePayload(EventPayload):
    turn: int
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    first_token_ms: Optional[int] = None
    elapsed_ms: int
    prompt_version: str
    finish_reason: Optional[str] = None


class ModelStartPayload(EventPayload):
    turn: int
    model: str
    message: str


class ModelToolCallPayload(EventPayload):
    turn: int
    call_id: str
    tool: str
    arguments: str


class FinalReportPayload(EventPayload):
    raw_report: str
    turns: int
    is_complete: bool
    status: Literal["success", "incomplete"]
    acknowledged_guardrails: list[str]
    finish_reason: Optional[str] = None


class ErrorPayload(EventPayload):
    error: str


class DonePayload(EventPayload):
    message: str


class PipelineStagePayload(EventPayload):
    stage: str
    status: Literal["started", "completed", "skipped", "failed"]
    message: str
    count: Optional[int] = None


class RuleAssessmentPayload(EventPayload):
    attempt: int
    raw_response: str
    rule_assessments: Any
    validated_assessments: Dict[str, Any]
    issues: list[str]


EVENT_PAYLOADS: Dict[str, Type[EventPayload]] = {
    "start": StartPayload,
    "token": TokenPayload,
    "report_token": ReportTokenPayload,
    "tool_start": ToolStartPayload,
    "tool_result": ToolResultPayload,
    "pipeline_stage": PipelineStagePayload,
    "rule_assessment": RuleAssessmentPayload,
    "guardrail": GuardrailPayload,
    "model_start": ModelStartPayload,
    "model_tool_call": ModelToolCallPayload,
    "model_usage": ModelUsagePayload,
    "final_report": FinalReportPayload,
    "error": ErrorPayload,
    "done": DonePayload,
}


def encode_event(name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    try:
        payload_model = EVENT_PAYLOADS[name]
    except KeyError as exc:
        raise ValueError(f"未定义的 SSE 事件: {name}") from exc
    validated = payload_model.model_validate(payload)
    return {"event": name, "data": validated.model_dump_json(exclude_none=True)}