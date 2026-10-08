"""Run the bounded ReAct model/tool interaction loop."""

import asyncio
import json
import logging
from time import perf_counter
from typing import Any, AsyncGenerator, Dict
from uuid import uuid4

from starlette.concurrency import iterate_in_threadpool

from configs.config import AGENT_CONFIG
from services.report_parser import clean_report_content, parse_structured_report

from .context_manager import ContextWindowError, ContextWindowManager
from .contracts import ToolExecutionResult
from .events import encode_event
from .final_report import (
    FinalReportValidationError,
    recover_final_report,
    validate_final_report,
)
from .guardrails import detect_contract_flags
from .prompt_builder import PROMPT_VERSION, build_review_messages
from .report_finalizer import ReportFinalizer
from .stream_decoder import ToolCallStreamDecoder
from .tool_catalog import FINAL_REPORT_TOOL_NAME, build_tool_schemas
from .tool_call_recorder import ToolCallRecorder
from .tool_executor import ToolExecutor


FINAL_REPORT_RETRY_LIMIT = 2
MODEL_STREAM_LOG_INTERVAL_SECONDS = 15
LOGGER = logging.getLogger("uvicorn.error")


async def _log_model_stream_progress(
    trace_id: str,
    turn: int,
    request_started: float,
    progress: Dict[str, int],
) -> None:
    while True:
        await asyncio.sleep(MODEL_STREAM_LOG_INTERVAL_SECONDS)
        LOGGER.info(
            "react_model_stream_progress trace_id=%s turn=%d elapsed_ms=%d "
            "chunks=%d reasoning_chars=%d content_chars=%d tool_call_deltas=%d",
            trace_id,
            turn,
            int((perf_counter() - request_started) * 1000),
            progress["chunks"],
            progress["reasoning_chars"],
            progress["content_chars"],
            progress["tool_call_deltas"],
        )


def _max_output_tokens(max_context_tokens: int, requested_output_tokens: int) -> int:
    return min(requested_output_tokens, max(1, max_context_tokens // 4))


class ReactLoop:
    """Run ReAct turns and publish model/tool interaction events."""

    def __init__(self, agent: Any):
        self.agent = agent

    @staticmethod
    def _without_enterprise_risk_levels(observation: str) -> str:
        try:
            payload = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            return observation
        if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), list):
            return observation
        for item in payload["evidence"]:
            if not isinstance(item, dict) or item.get("source_type") != "enterprise_rule":
                continue
            item.pop("enterprise_risk_level", None)
            for key in ("text", "snippet"):
                value = item.get(key)
                if not isinstance(value, str):
                    continue
                item[key] = "\n".join(
                    line for line in value.splitlines()
                    if not line.lstrip().startswith((
                        "风险级别:", "风险级别：", "风险等级:", "风险等级：",
                    ))
                )
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _enterprise_evidence_guidance(
        evidence: Dict[str, Dict[str, Any]],
    ) -> str:
        candidates = []
        for evidence_id, item in evidence.items():
            source_type = item.get("source_type")
            if source_type not in {"enterprise_rule", "enterprise_document"}:
                continue
            text = str(item.get("text") or "")
            text = "\n".join(
                line for line in text.splitlines()
                if not line.lstrip().startswith((
                    "风险级别:", "风险级别：", "风险等级:", "风险等级：",
                ))
            )
            reference_type = "RULE" if source_type == "enterprise_rule" else "KB"
            candidates.append({
                "source_type": source_type,
                "title": item.get("title") or item.get("doc_name"),
                "reference": f"[[{reference_type}:{evidence_id}]]",
                "text": text,
                **({
                    "enterprise_risk_level": item.get("enterprise_risk_level"),
                } if item.get("enterprise_risk_level") else {}),
            })
        candidates.sort(
            key=lambda item: item["source_type"] != "enterprise_rule"
        )
        records = []
        remaining_text_chars = 400
        for item in candidates[:8]:
            text_limit = min(
                remaining_text_chars,
                400 if item["source_type"] == "enterprise_rule" else 120,
            )
            record = {
                "type": "enterprise_rule" if item["source_type"] == "enterprise_rule" else "enterprise_document",
                "reference": item["reference"],
                "title": item["title"],
                "text": item["text"][:text_limit],
            }
            if item.get("enterprise_risk_level"):
                record["enterprise_risk_level"] = item["enterprise_risk_level"]
            records.append(record)
            remaining_text_chars -= min(len(item["text"]), text_limit)
        if not records:
            return ""
        return (
            "【必须遵守】以下企业证据由工具实际返回。合同涉及规则事项即适用，即使条款违反规则；"
            "enterprise_risk_level 是工具查到的企业内部风险等级，必须在报告中原样照录，不得改写、翻译或推断；"
            "企业依据须引用精确 reference，并依规则标准提出修改建议；法律依据与企业依据分开。"
            "不得称未检索到规则或编造编号。\n"
            + json.dumps(records, ensure_ascii=False)
        )

    async def stream(
        self,
        contract_text: str,
        max_turns: int,
        task_label: str,
        review_side: str = "neutral",
    ) -> AsyncGenerator[Dict[str, str], None]:
        unavailable = getattr(
            getattr(self.agent, "tool_registry", None),
            "unavailable_tools",
            set(),
        )
        findings = detect_contract_flags(contract_text)
        guardrail_codes = [finding.code for finding in findings]
        tools = build_tool_schemas(
            self.agent.tool_mapping,
            unavailable,
            guardrail_codes,
        )
        required_tool_names = [
            tool["function"]["name"]
            for tool in tools
            if tool["function"]["name"] != FINAL_REPORT_TOOL_NAME
        ]
        executor = ToolExecutor(self.agent.tool_mapping, unavailable)
        max_context_tokens = AGENT_CONFIG.get("max_context_tokens", 8192)
        max_output_tokens = _max_output_tokens(
            max_context_tokens,
            AGENT_CONFIG.get("max_tokens", 1024),
        )
        context = ContextWindowManager(
            model_name=self.agent.model_name,
            max_context_tokens=max_context_tokens,
            output_tokens=max_output_tokens,
        )
        messages = build_review_messages(contract_text, review_side, findings)
        duplicate_query_count = 0
        force_final_report = False
        final_report_retries = 0
        completed_turns = 0
        called_tool_names = set()
        recorder = ToolCallRecorder(task_label)
        recorder.start()
        turn_limit = max(max_turns, len(required_tool_names) + 1)
        trace_id = uuid4().hex[:12]
        stream_started = perf_counter()

        LOGGER.info(
            "react_started trace_id=%s model=%s review_side=%s turn_limit=%d "
            "required_tools=%d max_output_tokens=%d contract_chars=%d guardrails=%d",
            trace_id,
            self.agent.model_name,
            review_side,
            turn_limit,
            len(required_tool_names),
            max_output_tokens,
            len(contract_text),
            len(guardrail_codes),
        )

        yield encode_event("start", {
            "task_id": task_label,
            "message": "开始合同审查",
        })
        yield encode_event("pipeline_stage", {
            "task_id": task_label,
            "stage": "tool_recording",
            "status": "started",
            "message": "工具调用记录已开始",
        })
        if findings:
            yield encode_event("guardrail", {
                "task_id": task_label,
                "findings": [finding.model_dump() for finding in findings],
            })

        for turn in range(1, turn_limit + FINAL_REPORT_RETRY_LIMIT + 1):
            if turn > turn_limit + final_report_retries:
                break
            missing_tools = [
                name for name in required_tool_names
                if name not in called_tool_names
            ]
            if missing_tools:
                request_tools = [
                    tool for tool in tools
                    if tool["function"]["name"] == missing_tools[0]
                ]
            elif force_final_report:
                request_tools = [
                    tool for tool in tools
                    if tool["function"]["name"] == FINAL_REPORT_TOOL_NAME
                ]
            else:
                request_tools = tools
            try:
                request_source_messages = messages
                enterprise_evidence_pinned = False
                if not missing_tools:
                    enterprise_guidance = self._enterprise_evidence_guidance(
                        recorder.evidence_records()
                    )
                    if enterprise_guidance:
                        enterprise_evidence_pinned = True
                        request_source_messages = list(messages)
                        request_source_messages.append({
                            "role": "user",
                            "content": enterprise_guidance,
                        })
                request_messages = context.fit_messages(
                    request_source_messages,
                    request_tools,
                    enterprise_evidence_pinned=enterprise_evidence_pinned,
                )
            except ContextWindowError as exc:
                LOGGER.warning(
                    "react_context_window_error trace_id=%s turn=%d error=%s",
                    trace_id,
                    turn,
                    exc,
                )
                yield encode_event("error", {
                    "task_id": task_label,
                    "error": str(exc),
                })
                return

            decoder = ToolCallStreamDecoder()
            response = None
            request_started = perf_counter()
            first_token_ms = None
            progress = {
                "chunks": 0,
                "reasoning_chars": 0,
                "content_chars": 0,
                "tool_call_deltas": 0,
            }
            requested_tool_names = [
                tool["function"]["name"] for tool in request_tools
            ]
            LOGGER.info(
                "react_model_request_started trace_id=%s turn=%d/%d model=%s "
                "max_tokens=%d messages=%d input_chars=%d tools=%s force_final=%s",
                trace_id,
                turn,
                turn_limit,
                self.agent.model_name,
                max_output_tokens,
                len(request_messages),
                sum(len(str(message.get("content") or "")) for message in request_messages),
                ",".join(requested_tool_names),
                force_final_report,
            )
            yield encode_event("model_start", {
                "task_id": task_label,
                "turn": turn,
                "model": self.agent.model_name,
                "message": "正在请求模型生成响应；若本轮调用工具，模型可能不返回自然语言文本。",
            })
            heartbeat_task = asyncio.create_task(
                _log_model_stream_progress(
                    trace_id,
                    turn,
                    request_started,
                    progress,
                )
            )
            request_options = {
                "model": self.agent.model_name,
                "messages": request_messages,
                "temperature": AGENT_CONFIG.get("temperature", 0.0),
                "top_p": AGENT_CONFIG.get("top_p", 1.0),
                "seed": AGENT_CONFIG.get("seed", 42),
                "max_tokens": max_output_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if request_tools:
                if missing_tools:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": missing_tools[0]},
                    }
                else:
                    tool_choice = "required" if force_final_report else "auto"
                request_options.update({
                    "tools": request_tools,
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": False,
                })

            try:
                response = await asyncio.to_thread(
                    self.agent.client.chat.completions.create,
                    **request_options,
                )
                async for chunk in iterate_in_threadpool(response):
                    progress["chunks"] += 1
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        reasoning = (
                            getattr(delta, "reasoning", None)
                            or getattr(delta, "reasoning_content", None)
                        ) if delta else None
                        if reasoning:
                            progress["reasoning_chars"] += len(reasoning)
                        progress["tool_call_deltas"] += len(
                            getattr(delta, "tool_calls", None) or []
                        ) if delta else 0
                    text, report_deltas = decoder.feed(chunk)
                    if text or report_deltas:
                        progress["content_chars"] += len(text or "")
                        progress["content_chars"] += sum(
                            len(item) for item in report_deltas
                        )
                        if first_token_ms is None:
                            first_token_ms = int((perf_counter() - request_started) * 1000)
                    if text:
                        yield encode_event("token", {
                            "task_id": task_label,
                            "token": text,
                        })
                    for report_delta in report_deltas:
                        yield encode_event("report_token", {
                            "task_id": task_label,
                            "token": report_delta,
                        })
            except asyncio.CancelledError:
                LOGGER.info(
                    "react_model_request_cancelled trace_id=%s turn=%d elapsed_ms=%d",
                    trace_id,
                    turn,
                    int((perf_counter() - request_started) * 1000),
                )
                raise
            except Exception as exc:
                LOGGER.exception(
                    "react_model_request_failed trace_id=%s turn=%d elapsed_ms=%d "
                    "error_type=%s",
                    trace_id,
                    turn,
                    int((perf_counter() - request_started) * 1000),
                    type(exc).__name__,
                )
                yield encode_event("error", {
                    "task_id": task_label,
                    "error": f"大模型交互失败: {exc}",
                })
                return
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                close = getattr(response, "close", None)
                if close:
                    try:
                        await asyncio.to_thread(close)
                    except Exception:
                        LOGGER.warning(
                            "react_model_response_close_failed trace_id=%s turn=%d",
                            trace_id,
                            turn,
                            exc_info=True,
                        )

            elapsed_ms = int((perf_counter() - request_started) * 1000)
            completed_turns = turn
            content, tool_calls = decoder.result()
            prompt_tokens = decoder.prompt_tokens or context.estimate_tokens({
                "messages": request_messages,
                "tools": request_tools,
            })
            completion_tokens = decoder.completion_tokens or context.estimate_tokens({
                "content": content,
                "tool_calls": [call.model_dump() for call in tool_calls],
            })
            LOGGER.info(
                "react_model_request_completed trace_id=%s turn=%d elapsed_ms=%d "
                "first_content_ms=%s finish_reason=%s prompt_tokens=%d "
                "completion_tokens=%d usage_source=%s reasoning_chars=%d "
                "content_chars=%d chunks=%d tool_calls=%d",
                trace_id,
                turn,
                elapsed_ms,
                first_token_ms,
                decoder.finish_reason or "none",
                prompt_tokens,
                completion_tokens,
                "model" if decoder.prompt_tokens is not None
                and decoder.completion_tokens is not None else "estimated",
                progress["reasoning_chars"],
                progress["content_chars"],
                progress["chunks"],
                len(tool_calls),
            )
            yield encode_event("model_usage", {
                "task_id": task_label,
                "turn": turn,
                "model": self.agent.model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "first_token_ms": first_token_ms,
                "elapsed_ms": elapsed_ms,
                "prompt_version": PROMPT_VERSION,
                "finish_reason": decoder.finish_reason,
            })

            for call in tool_calls:
                LOGGER.info(
                    "react_tool_call_received trace_id=%s turn=%d tool=%s "
                    "call_id=%s argument_chars=%d",
                    trace_id,
                    turn,
                    call.name,
                    call.call_id,
                    len(call.arguments),
                )
                recorder.record_call(
                    turn,
                    call.call_id,
                    call.name,
                    call.arguments,
                )
                yield encode_event("model_tool_call", {
                    "task_id": task_label,
                    "turn": turn,
                    "call_id": call.call_id,
                    "tool": call.name,
                    "arguments": call.arguments,
                })

            if tool_calls:
                called_tool_names.update(
                    call.name for call in tool_calls
                    if call.name in required_tool_names
                )
                assistant_tool_message = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
                messages.append(assistant_tool_message)

                for call in tool_calls:
                    started = perf_counter()
                    if call.name == FINAL_REPORT_TOOL_NAME:
                        missing_tools = [
                            name for name in required_tool_names
                            if name not in called_tool_names
                        ]
                        if missing_tools:
                            error = (
                                "尚未调用所有必需检索工具："
                                + ", ".join(missing_tools)
                            )
                            observation = json.dumps({
                                "ok": False,
                                "error": error,
                                "instruction": (
                                    "先逐个调用每个未完成的只读检索工具，再提交最终报告。"
                                ),
                            }, ensure_ascii=False)
                            result = ToolExecutionResult(
                                call_id=call.call_id,
                                tool_name=call.name,
                                success=False,
                                observation=observation,
                                error=error,
                                elapsed_ms=int((perf_counter() - started) * 1000),
                            )
                            force_final_report = True
                            if final_report_retries < FINAL_REPORT_RETRY_LIMIT:
                                final_report_retries += 1
                                LOGGER.warning(
                                    "react_empty_model_response trace_id=%s turn=%d "
                                    "finish_reason=%s retry=%d/%d reasoning_chars=%d",
                                    trace_id,
                                    turn,
                                    decoder.finish_reason or "none",
                                    final_report_retries,
                                    FINAL_REPORT_RETRY_LIMIT,
                                    progress["reasoning_chars"],
                                )
                            yield encode_event("tool_start", {
                                "task_id": task_label,
                                "tool": call.name,
                                "call_id": call.call_id,
                                "query": None,
                            })
                            yield encode_event("tool_result", {
                                "task_id": task_label,
                                "tool": call.name,
                                "call_id": call.call_id,
                                "observation": result.observation,
                                "success": False,
                                "elapsed_ms": result.elapsed_ms,
                                "error": error,
                            })
                            recorder.record_result(
                                call.call_id,
                                result.observation,
                                result.success,
                                result.elapsed_ms,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "name": call.name,
                                "content": result.observation,
                            })
                            continue
                        try:
                            submission = validate_final_report(
                                call,
                                guardrail_codes,
                                decoder.finish_reason,
                                contract_text,
                            )
                        except FinalReportValidationError as exc:
                            LOGGER.warning(
                                "react_final_report_rejected trace_id=%s turn=%d "
                                "reason=%s issue_count=%d issue_types=%s "
                                "finish_reason=%s retry=%d/%d rejected_arguments=%s",
                                trace_id,
                                turn,
                                exc.reason_code,
                                exc.issue_count,
                                ",".join(exc.issue_types) or "none",
                                decoder.finish_reason or "none",
                                final_report_retries,
                                FINAL_REPORT_RETRY_LIMIT,
                                json.dumps(call.arguments, ensure_ascii=False),
                            )
                            fallback = recover_final_report(call, guardrail_codes)
                            if fallback is not None:
                                submission, normalized_fields, dropped_reviews = fallback
                                LOGGER.warning(
                                    "react_final_report_fallback trace_id=%s turn=%d "
                                    "reason=%s normalized_fields=%d dropped_reviews=%d",
                                    trace_id,
                                    turn,
                                    exc.reason_code,
                                    normalized_fields,
                                    dropped_reviews,
                                )
                                recorder.record_result(
                                    call.call_id,
                                    json.dumps(
                                        {"report": submission.report.model_dump(mode="json")},
                                        ensure_ascii=False,
                                    ),
                                    True,
                                    int((perf_counter() - started) * 1000),
                                )
                                finalizer = ReportFinalizer(self.agent)
                                async for event in finalizer.finalize(
                                    report=submission.report.model_dump_json(
                                        ensure_ascii=False
                                    ),
                                    contract_text=contract_text,
                                    task_id=task_label,
                                    turn=turn,
                                    acknowledged_guardrails=submission.acknowledged_guardrails,
                                    finish_reason=decoder.finish_reason,
                                    tool_call_records=recorder,
                                ):
                                    yield event
                                return
                            force_final_report = True
                            if final_report_retries < FINAL_REPORT_RETRY_LIMIT:
                                final_report_retries += 1
                            LOGGER.warning(
                                "react_final_report_fallback_unavailable "
                                "trace_id=%s turn=%d reason=%s",
                                trace_id,
                                turn,
                                exc.reason_code,
                            )
                            observation = json.dumps({
                                "ok": False,
                                "error": str(exc),
                                "instruction": (
                                    "只重新提交 submit_final_report。acknowledged_guardrails 只能包含本次程序规则代码，"
                                    "不得填写工具名、证据编号或法规编号；没有程序规则代码时传空数组。"
                                    "如果错误指出遗漏条款，必须补齐对应审查项；没有实质风险时如实说明，不得虚构风险。"
                                ),
                            }, ensure_ascii=False)
                            result = ToolExecutionResult(
                                call_id=call.call_id,
                                tool_name=call.name,
                                success=False,
                                observation=observation,
                                error=str(exc),
                                elapsed_ms=int((perf_counter() - started) * 1000),
                            )
                        else:
                            if len(tool_calls) != 1:
                                force_final_report = True
                                if final_report_retries < FINAL_REPORT_RETRY_LIMIT:
                                    final_report_retries += 1
                                LOGGER.warning(
                                    "react_final_report_rejected trace_id=%s turn=%d "
                                    "reason=multiple_final_report_tool_calls "
                                    "tool_call_count=%d retry=%d/%d "
                                    "rejected_arguments=%s",
                                    trace_id,
                                    turn,
                                    len(tool_calls),
                                    final_report_retries,
                                    FINAL_REPORT_RETRY_LIMIT,
                                    json.dumps(call.arguments, ensure_ascii=False),
                                )
                                LOGGER.warning(
                                    "react_final_report_fallback trace_id=%s turn=%d "
                                    "reason=multiple_final_report_tool_calls "
                                    "normalized_fields=0 dropped_reviews=0",
                                    trace_id,
                                    turn,
                                )
                                recorder.record_result(
                                    call.call_id,
                                    json.dumps(
                                        {"report": submission.report.model_dump(mode="json")},
                                        ensure_ascii=False,
                                    ),
                                    True,
                                    int((perf_counter() - started) * 1000),
                                )
                                finalizer = ReportFinalizer(self.agent)
                                async for event in finalizer.finalize(
                                    report=submission.report.model_dump_json(
                                        ensure_ascii=False
                                    ),
                                    contract_text=contract_text,
                                    task_id=task_label,
                                    turn=turn,
                                    acknowledged_guardrails=submission.acknowledged_guardrails,
                                    finish_reason=decoder.finish_reason,
                                    tool_call_records=recorder,
                                ):
                                    yield event
                                return
                            else:
                                recorder.record_result(
                                    call.call_id,
                                    json.dumps(
                                        {"report": submission.report.model_dump(mode="json")},
                                        ensure_ascii=False,
                                    ),
                                    True,
                                    int((perf_counter() - started) * 1000),
                                )
                                LOGGER.info(
                                    "react_final_report_accepted trace_id=%s turn=%d "
                                    "finish_reason=%s elapsed_ms=%d total_elapsed_ms=%d",
                                    trace_id,
                                    turn,
                                    decoder.finish_reason or "none",
                                    elapsed_ms,
                                    int((perf_counter() - stream_started) * 1000),
                                )
                                finalizer = ReportFinalizer(self.agent)
                                async for event in finalizer.finalize(
                                    report=submission.report.model_dump_json(
                                        ensure_ascii=False
                                    ),
                                    contract_text=contract_text,
                                    task_id=task_label,
                                    turn=turn,
                                    acknowledged_guardrails=submission.acknowledged_guardrails,
                                    finish_reason=decoder.finish_reason,
                                    tool_call_records=recorder,
                                ):
                                    yield event
                                return
                    else:
                        result = await asyncio.to_thread(executor.execute, call)
                        if result.duplicate:
                            duplicate_query_count += 1
                            if duplicate_query_count >= 2:
                                force_final_report = True

                    LOGGER.info(
                        "react_tool_call_completed trace_id=%s turn=%d tool=%s "
                        "call_id=%s success=%s duplicate=%s elapsed_ms=%d "
                        "observation_chars=%d",
                        trace_id,
                        turn,
                        call.name,
                        call.call_id,
                        result.success,
                        result.duplicate,
                        result.elapsed_ms,
                        len(result.observation),
                    )

                    query = executor.query_preview(call)
                    yield encode_event("tool_start", {
                        "task_id": task_label,
                        "tool": call.name,
                        "call_id": call.call_id,
                        "query": query,
                    })
                    model_observation = result.observation
                    if result.success:
                        model_observation = self._without_enterprise_risk_levels(
                            model_observation
                        )
                    tool_content = (
                        context.compact_observation(model_observation)
                        if result.success else result.observation
                    )
                    recorder.record_result(
                        call.call_id,
                        result.observation,
                        result.success,
                        result.elapsed_ms,
                    )
                    result_evidence = recorder.evidence_for_call(call.call_id)
                    yield encode_event("tool_result", {
                        "task_id": task_label,
                        "tool": call.name,
                        "call_id": call.call_id,
                        "observation": result.observation,
                        "success": result.success,
                        "elapsed_ms": result.elapsed_ms,
                        "injected_chars": len(tool_content) if result.success else 0,
                        "evidence_sources": [
                            {
                                "evidence_id": evidence_id,
                                "source_type": item.get("source_type", "unknown"),
                                "source_location": (
                                    item.get("source_location")
                                    or item.get("source_path")
                                    or item.get("doc_name")
                                    or "来源位置未提供"
                                ),
                            }
                            for evidence_id, item in result_evidence.items()
                        ],
                        "error": result.error,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": tool_content,
                    })
                continue

            if not content:
                if final_report_retries < FINAL_REPORT_RETRY_LIMIT:
                    final_report_retries += 1
                    if missing_tools:
                        retry_instruction = (
                            "上一轮没有返回有效工具调用。请立即调用必需工具 "
                            f"{missing_tools[0]}，不要只输出分析。"
                        )
                    else:
                        force_final_report = True
                        retry_instruction = (
                            "上一轮没有返回有效工具调用或最终文本。请立即单独调用 "
                            "submit_final_report 提交最终报告，不要只输出分析。"
                        )
                    messages.extend([
                        {"role": "assistant", "content": ""},
                        {"role": "user", "content": retry_instruction},
                    ])
                    continue
                finish_reason = (
                    f"（finish_reason={decoder.finish_reason}）"
                    if decoder.finish_reason else ""
                )
                LOGGER.error(
                    "react_empty_model_response_exhausted trace_id=%s turn=%d "
                    "finish_reason=%s total_elapsed_ms=%d",
                    trace_id,
                    turn,
                    decoder.finish_reason or "none",
                    int((perf_counter() - stream_started) * 1000),
                )
                yield encode_event("error", {
                    "task_id": task_label,
                    "error": f"模型未返回文本或有效工具调用{finish_reason}。",
                })
                return
            final_report_text = clean_report_content(content)
            parsed_report = parse_structured_report(final_report_text)
            if parsed_report:
                LOGGER.info(
                    "react_text_report_accepted trace_id=%s turn=%d total_elapsed_ms=%d",
                    trace_id,
                    turn,
                    int((perf_counter() - stream_started) * 1000),
                )
                finalizer = ReportFinalizer(self.agent)
                async for event in finalizer.finalize(
                    report=parsed_report.model_dump_json(ensure_ascii=False),
                    contract_text=contract_text,
                    task_id=task_label,
                    turn=turn,
                    acknowledged_guardrails=guardrail_codes,
                    finish_reason=decoder.finish_reason,
                    tool_call_records=recorder,
                ):
                    yield event
                return
            force_final_report = True
            messages.extend([
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "最终报告必须通过 submit_final_report 工具提交，"
                        "请按其 JSON Schema 提供 report 和 acknowledged_guardrails。"
                    ),
                },
            ])

        yield encode_event("final_report", {
            "task_id": task_label,
            "raw_report": "",
            "turns": completed_turns,
            "is_complete": False,
            "status": "incomplete",
            "acknowledged_guardrails": [],
        })
        LOGGER.error(
            "react_turn_limit_reached trace_id=%s completed_turns=%d turn_limit=%d "
            "final_report_retries=%d total_elapsed_ms=%d",
            trace_id,
            completed_turns,
            turn_limit,
            final_report_retries,
            int((perf_counter() - stream_started) * 1000),
        )
        yield encode_event("error", {
            "task_id": task_label,
            "error": (
                f"达到最大交互轮数 ({turn_limit}) 或最终报告校正重试上限，"
                "模型尚未提交通过校验的最终报告。"
            ),
        })