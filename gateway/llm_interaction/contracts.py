"""Validated data contracts for model tool calls and execution results."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryArguments(StrictModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="用于检索的具体关键词或问题",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class ModelToolCall(StrictModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str


class FinalReportArguments(StrictModel):
    report: str = Field(min_length=1, max_length=30000)
    acknowledged_guardrails: list[str] = Field(max_length=8)

    @field_validator("report")
    @classmethod
    def strip_report(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("report 不能为空")
        return value

    @field_validator("acknowledged_guardrails")
    @classmethod
    def unique_guardrails(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("acknowledged_guardrails 不得包含重复代码")
        return value


class ToolExecutionResult(StrictModel):
    call_id: str
    tool_name: str
    query: Optional[str] = None
    success: bool
    duplicate: bool = False
    observation: str
    error: Optional[str] = None
    elapsed_ms: int = Field(ge=0)


class GuardrailFinding(StrictModel):
    code: str
    severity: str
    message: str
    evidence: str