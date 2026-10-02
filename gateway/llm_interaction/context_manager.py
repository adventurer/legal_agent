"""Token budgeting and sliding-window compaction for ReAct history."""

import json
from typing import Any, Dict, List

import tiktoken


INPUT_CONTEXT_RATIO = 0.98


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
            int(max_context_tokens * INPUT_CONTEXT_RATIO) - output_tokens,
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
        enterprise_evidence_pinned: bool = False,
    ) -> List[Dict[str, Any]]:
        fixed_messages = messages[:2]
        history = messages[2:]
        trailing_messages = []
        if history and history[-1].get("role") in {"system", "user"}:
            trailing_messages = [history[-1]]
            history = history[:-1]
        all_turns = self._group_tool_turns(history)
        evidence_turns = [] if enterprise_evidence_pinned else [
            turn for turn in all_turns
            if self._contains_enterprise_evidence(turn)
        ]
        evidence_turn_ids = {id(turn) for turn in evidence_turns}
        other_turns = [
            turn for turn in all_turns
            if id(turn) not in evidence_turn_ids
        ]
        remaining_slots = max(0, self._max_history_turns - len(evidence_turns))
        turns = sorted(
            evidence_turns + (other_turns[-remaining_slots:] if remaining_slots else []),
            key=all_turns.index,
        )

        while turns and self._estimate_request(
            fixed_messages + self._flatten(turns) + trailing_messages,
            tools,
        ) > self._max_input_tokens:
            removable_index = next(
                (
                    index for index, turn in enumerate(turns)
                    if id(turn) not in evidence_turn_ids
                ),
                0,
            )
            turns.pop(removable_index)

        fitted = fixed_messages + self._flatten(turns) + trailing_messages
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
    def _contains_enterprise_evidence(turn: List[Dict[str, Any]]) -> bool:
        for message in turn:
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            evidence = payload.get("evidence") if isinstance(payload, dict) else None
            if isinstance(evidence, list) and any(
                isinstance(item, dict)
                and item.get("source_type") in {
                    "enterprise_document", "enterprise_rule",
                }
                for item in evidence
            ):
                return True
        return False

    @staticmethod
    def _flatten(turns: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [message for turn in turns for message in turn]