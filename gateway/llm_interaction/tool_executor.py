"""Validate and execute approved read-only model tool calls."""

import json
import unicodedata
from time import perf_counter
from typing import Any, Callable, Dict, Iterable

from pydantic import ValidationError

from .contracts import ModelToolCall, QueryArguments, ToolExecutionResult
from .tool_catalog import READ_ONLY_TOOL_DESCRIPTIONS


class ToolExecutor:
    def __init__(
        self,
        tool_mapping: Dict[str, Callable[[str], Any]],
        unavailable_tools: Iterable[str] = (),
    ) -> None:
        unavailable = set(unavailable_tools)
        self._tools = {
            name: tool
            for name, tool in tool_mapping.items()
            if name in READ_ONLY_TOOL_DESCRIPTIONS and name not in unavailable
        }
        self._executed_queries: set[tuple[str, str]] = set()

    @staticmethod
    def _query_key(tool_name: str, query: str) -> tuple[str, str]:
        normalized = unicodedata.normalize("NFKC", query).casefold()
        normalized = "".join(
            character
            for character in normalized
            if not character.isspace()
            and unicodedata.category(character)[0] not in {"P", "S"}
        )
        return tool_name, normalized

    @staticmethod
    def query_preview(call: ModelToolCall) -> str | None:
        try:
            return QueryArguments.model_validate_json(call.arguments).query
        except (ValidationError, ValueError, TypeError):
            return None

    def execute(self, call: ModelToolCall) -> ToolExecutionResult:
        started = perf_counter()
        query = None
        try:
            tool = self._tools.get(call.name)
            if tool is None:
                raise ValueError(f"工具不可用或未获准执行: {call.name}")
            arguments = QueryArguments.model_validate_json(call.arguments)
            query = arguments.query
            query_key = self._query_key(call.name, query)
            if query_key in self._executed_queries:
                raise ValueError(
                    "重复检索：该工具已执行过相同查询。请改用具体且不同的关键词，"
                    "或根据已有检索结果提交最终报告。"
                )
            result = tool(query)
            self._executed_queries.add(query_key)
            observation = result if isinstance(result, str) else json.dumps(
                result, ensure_ascii=False, default=str
            )
            return ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                query=query,
                success=True,
                observation=observation,
                elapsed_ms=int((perf_counter() - started) * 1000),
            )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            observation = json.dumps({
                "ok": False,
                "error": error,
                "instruction": (
                    "请使用具体且不同的检索关键词，或根据已有结果提交最终报告。"
                    if error.startswith("重复检索：")
                    else "请修正工具参数或选择可用工具后重试。"
                ),
            }, ensure_ascii=False)
            return ToolExecutionResult(
                call_id=call.call_id,
                tool_name=call.name,
                query=query,
                success=False,
                duplicate=error.startswith("重复检索："),
                observation=observation,
                error=error,
                elapsed_ms=int((perf_counter() - started) * 1000),
            )