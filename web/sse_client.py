"""合同审查 SSE 客户端。"""

import json
from typing import Any, Dict, Iterator

import httpx
from httpx_sse import connect_sse


def stream_contract_review(
    base_url: str,
    contract_text: str,
    max_turns: int,
    timeout: float = 180.0,
    debug: bool = False,
) -> Iterator[Dict[str, Any]]:
    """请求合同审查流，并将 SSE 事件转换为字典。"""
    with httpx.Client(timeout=timeout) as client:
        payload = {
            "contract_text": contract_text,
            "max_turns": max_turns,
            "stream": True,
            "debug": debug,
        }
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
