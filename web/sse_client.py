"""合同审查 SSE 客户端。"""

import json
from typing import Any, Dict, Iterator

import httpx
from httpx_sse import connect_sse


def stream_contract_review(
    base_url: str,
    contract_text: str,
    max_turns: int,
    review_run_id: str | None = None,
    review_side: str = "neutral",
    timeout: float = 180.0,
) -> Iterator[Dict[str, Any]]:
    """请求合同审查流，并将 SSE 事件转换为字典。"""
    with httpx.Client(timeout=timeout) as client:
        payload = {
            "contract_text": contract_text,
            "max_turns": max_turns,
            "stream": True,
            "review_side": review_side,
        }
        if review_run_id:
            payload["review_run_id"] = review_run_id
        with connect_sse(
            client,
            "POST",
            f"{base_url}/api/v1/contract/review/stream",
            json=payload,
        ) as event_source:
            for sse in event_source.iter_sse():
                yield {
                    "event": sse.event,
                    "data": json.loads(sse.data) if sse.data else {},
                }
