"""Append-only local traces for reproducing contract review runs."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_TRACE_LOCK = threading.Lock()
_INITIALIZED_RUNS = set()


class ReviewTrace:
    """Persist the exact submitted contract and review events to a JSONL file."""

    def __init__(
        self,
        trace_dir: Path,
        contract_text: str,
        max_turns: int,
        model: str,
        task_label: str,
        review_run_id: Optional[str] = None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.review_run_id = review_run_id or uuid.uuid4().hex
        self.path = trace_dir / "review_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_LOCK:
            if self.review_run_id not in _INITIALIZED_RUNS:
                self.path.write_text("", encoding="utf-8")
                _INITIALIZED_RUNS.add(self.review_run_id)
            self._file = self.path.open("a", encoding="utf-8")
        self._assistant_text = ""
        self._closed = False
        self._write({
            "type": "review_start",
            "trace_id": self.trace_id,
            "review_run_id": self.review_run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "task_label": task_label,
            "model": model,
            "max_turns": max_turns,
            "contract_text": contract_text,
        })

    def _write(self, record: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            record.setdefault("trace_id", self.trace_id)
            record.setdefault("review_run_id", self.review_run_id)
            with _TRACE_LOCK:
                self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._file.flush()
                os.fsync(self._file.fileno())
        except OSError as exc:
            print(f"[警告] 写入审查调试记录失败 ({self.trace_id}): {exc}", flush=True)
            self._closed = True
            try:
                self._file.close()
            except OSError:
                pass

    def record_sse(self, item: Dict[str, Any]) -> None:
        try:
            event = item.get("event", "unknown")
            payload = json.loads(item.get("data", "{}"))
        except (TypeError, ValueError):
            event, payload = "unknown", {"raw_data": str(item)}

        if event == "token":
            self._assistant_text += str(payload.get("token", ""))
            return

        if self._assistant_text:
            self._write({
                "type": "model_output",
                "text": self._assistant_text,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            self._assistant_text = ""

        self._write({
            "type": event,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        })

    def close(self, status: str = "stream_closed") -> None:
        if self._closed:
            return
        if self._assistant_text:
            self._write({
                "type": "model_output_partial",
                "text": self._assistant_text,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            self._assistant_text = ""
        self._write({
            "type": "review_end",
            "status": status,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        self._closed = True
        try:
            self._file.close()
        except OSError:
            pass
