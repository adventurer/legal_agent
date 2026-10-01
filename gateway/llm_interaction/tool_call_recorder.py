"""Capture one review's tool calls and their returned knowledge evidence."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolCallRecorder:
    """Keep a structured in-memory ledger for a single review run."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self.status: Optional[str] = None
        self._calls: List[Dict[str, Any]] = []

    def start(self) -> None:
        self.started_at = _timestamp()

    def record_call(
        self,
        turn: int,
        call_id: str,
        tool: str,
        arguments: str,
    ) -> None:
        self._calls.append({
            "turn": turn,
            "call_id": call_id,
            "tool": tool,
            "arguments": arguments,
            "started_at": _timestamp(),
            "success": None,
            "elapsed_ms": None,
            "observation": None,
            "evidence": [],
        })

    def record_result(
        self,
        call_id: str,
        observation: str,
        success: bool,
        elapsed_ms: int,
    ) -> None:
        record = next(
            (
                item for item in reversed(self._calls)
                if item["call_id"] == call_id and item["success"] is None
            ),
            None,
        )
        if record is None:
            return
        record.update({
            "success": success,
            "elapsed_ms": elapsed_ms,
            "observation": observation,
            "completed_at": _timestamp(),
        })
        try:
            payload = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            return
        evidence = payload.get("evidence", []) if isinstance(payload, dict) else []
        if isinstance(evidence, list):
            record["evidence"] = [
                item for item in evidence
                if isinstance(item, dict) and item.get("id")
            ]

    def finish(self, status: str) -> None:
        self.ended_at = _timestamp()
        self.status = status

    def evidence_records(self) -> Dict[str, Dict[str, Any]]:
        return {
            item["id"]: item
            for call in self._calls
            for item in call["evidence"]
        }

    def evidence_for_call(self, call_id: str) -> Dict[str, Dict[str, Any]]:
        return {
            item["id"]: item
            for call in self._calls
            if call["call_id"] == call_id
            for item in call["evidence"]
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "calls": [dict(call) for call in self._calls],
        }
