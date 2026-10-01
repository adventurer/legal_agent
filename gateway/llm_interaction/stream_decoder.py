"""Assemble OpenAI-compatible content and tool-call streaming deltas."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from partial_json_parser import Allow, loads as load_partial_json

from .contracts import ModelToolCall


class ToolCallStreamDecoder:
    def __init__(self) -> None:
        self.content_parts: List[str] = []
        self._tool_calls: Dict[int, Dict[str, str]] = {}
        self.finish_reason: Optional[str] = None
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self._streamed_reports: Dict[int, str] = {}

    def feed(self, chunk: Any) -> Tuple[Optional[str], List[str]]:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.prompt_tokens = getattr(usage, "prompt_tokens", None)
            self.completion_tokens = getattr(usage, "completion_tokens", None)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return None, []
        choice = choices[0]
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            self.content_parts.append(content)

        report_deltas = []
        for fallback_index, call_delta in enumerate(
            getattr(delta, "tool_calls", None) or []
        ):
            index = getattr(call_delta, "index", None)
            if index is None:
                index = fallback_index
            state = self._tool_calls.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )
            state["id"] += getattr(call_delta, "id", None) or ""
            function = getattr(call_delta, "function", None)
            if function:
                state["name"] += getattr(function, "name", None) or ""
                state["arguments"] += getattr(function, "arguments", None) or ""
            if state["name"] == "submit_final_report":
                try:
                    partial = load_partial_json(state["arguments"], Allow.ALL)
                except (ValueError, TypeError):
                    continue
                report = partial.get("report") if isinstance(partial, dict) else None
                previous = self._streamed_reports.get(index, "")
                if isinstance(report, str) and report.startswith(previous):
                    report_delta = report[len(previous):]
                    self._streamed_reports[index] = report
                    if report_delta:
                        report_deltas.append(report_delta)
        return content, report_deltas

    def result(self) -> Tuple[str, List[ModelToolCall]]:
        calls = [
            ModelToolCall(
                call_id=state["id"] or f"call_{uuid.uuid4().hex}",
                name=state["name"] or "__invalid_tool_call__",
                arguments=state["arguments"],
            )
            for _, state in sorted(self._tool_calls.items())
        ]
        return "".join(self.content_parts).strip(), calls