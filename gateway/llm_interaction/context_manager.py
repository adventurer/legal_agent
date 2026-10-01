"""Token budgeting and sliding-window compaction for ReAct history."""

import json
from typing import Any, Dict, List

import tiktoken


class ContextWindowError(ValueError):
    """Raised when fixed request content alone exceeds the model window."""


class ContextWindowManager:
    def __init__(
        self,
        model_name: str,
        max_context_tokens: int,
        output_tokens: int,
        max_history_turns: int = 4,
        max_observation_chars: int = 2400,
    ) -> None:
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self._max_input_tokens = max(
            128,
            int(max_context_tokens * 0.92) - output_tokens,
        )
        self._max_history_turns = max_history_turns
        self._max_observation_chars = max_observation_chars

    def estimate_tokens(self, value: Any) -> int:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        return len(self._encoding.encode(serialized))

    def compact_observation(self, observation: str) -> str:
        try:
            payload = json.loads(observation)
        except (TypeError, json.JSONDecodeError):
            return observation[:self._max_observation_chars]

        if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            payload["evidence"] = payload["evidence"][:5]
            for item in payload["evidence"]:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and len(text) > self._max_observation_chars:
                    item["text"] = text[:self._max_observation_chars] + "…"
        return json.dumps(payload, ensure_ascii=False, default=str)

    def fit_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fixed_messages = messages[:2]
        history = messages[2:]
        turns = self._group_tool_turns(history)
        turns = turns[-self._max_history_turns:]

        while turns and self._estimate_request(fixed_messages + self._flatten(turns), tools) > self._max_input_tokens:
            turns.pop(0)

        fitted = fixed_messages + self._flatten(turns)
        estimated = self._estimate_request(fitted, tools)
        if estimated > self._max_input_tokens:
            raise ContextWindowError(
                "合同正文和基础提示已超出当前模型可用输入窗口 "
                f"({estimated} > {self._max_input_tokens} tokens)"
            )
        return fitted

    def _estimate_request(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> int:
        return self.estimate_tokens({"messages": messages, "tools": tools}) + 4 * len(messages)

    @staticmethod
    def _group_tool_turns(history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        turns: List[List[Dict[str, Any]]] = []
        for message in history:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                turns.append([message])
            elif message.get("role") == "tool" and turns:
                turns[-1].append(message)
        return turns

    @staticmethod
    def _flatten(turns: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [message for turn in turns for message in turn]