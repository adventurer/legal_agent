"""Streaming contract-review orchestration for the gateway."""

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List

from configs.config import AGENT_CONFIG
from configs.kb_recommendations import (
    DEFAULT_RECOMMENDATION,
    DEFAULT_RECOMMENDATION_QUERY_LIMIT,
    FALLBACK_RECOMMENDATION_TEMPLATE,
    KB_RECOMMENDATION_RULES,
)
from core.action_parser import extract_action, extract_actions
from core.prompts import (
    AGENT_SYSTEM_PROMPT,
    EMPTY_SEARCH_RETRY_PROMPT,
    FORCE_FINAL_CONVERGENCE_PROMPT,
    TOOL_CALL_RETRY_PROMPT,
    USER_CONTRACT_INPUT_TEMPLATE,
    format_observation,
)
from services.report_parser import clean_report_content, is_final_report

MODEL_MAX_CONTEXT = AGENT_CONFIG.get("max_context_tokens", 8192)
CONTEXT_THRESHOLD_95 = int(MODEL_MAX_CONTEXT * 0.95)
OUTPUT_MAX_TOKENS = 4096
REASONING_MAX_TOKENS = AGENT_CONFIG.get("max_tokens", 1024)
CONTEXT_SAFETY_RATIO = 0.95


def estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate input tokens conservatively when the model tokenizer is unavailable."""
    text = "\n".join(str(message.get("content", "")) for message in messages)
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except (ImportError, LookupError, ValueError):
        cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        non_cjk_text = re.sub(r"[\u3400-\u9fff]", " ", text)
        non_cjk_tokens = len(re.findall(r"\w+|[^\s\w]", non_cjk_text))
        return cjk_chars + non_cjk_tokens


def context_budget_exceeded(input_tokens: int, output_budget: int) -> bool:
    """Check whether the requested input plus output budget reaches the safe limit."""
    return input_tokens + output_budget >= int(MODEL_MAX_CONTEXT * CONTEXT_SAFETY_RATIO)


def generate_kb_deficit_recommendation(
    attempted_queries: List[str], text_sample: str
) -> List[str]:
    """Generate concrete legal sources suggested by failed or exhausted searches."""
    recommendations = []
    combined_context = (" ".join(attempted_queries) + " " + text_sample).lower()

    for rule in KB_RECOMMENDATION_RULES:
        keywords = rule.get("keywords", [])
        if any(keyword.lower() in combined_context for keyword in keywords):
            recommendations.append(str(rule["recommendation"]))

    if not recommendations:
        clean_terms = [q for q in attempted_queries if q.strip()]
        terms_str = (
            "、".join(
                f"'{term}'"
                for term in clean_terms[:DEFAULT_RECOMMENDATION_QUERY_LIMIT]
            )
            if clean_terms
            else DEFAULT_RECOMMENDATION
        )
        recommendations.append(FALLBACK_RECOMMENDATION_TEMPLATE.format(terms=terms_str))
    return recommendations


def _event(name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


class ReviewOrchestrator:
    """Owns the SSE ReAct loop while leaving HTTP concerns in api_server."""

    def __init__(self, agent: Any, debug: bool = False):
        self.agent = agent
        self.debug = debug

    async def stream(
        self, contract_text: str, max_turns: int, task_label: str
    ) -> AsyncGenerator[Dict[str, str], None]:
        debug = self.debug
        max_search_budget = max(3, max_turns - 1)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTRACT_INPUT_TEMPLATE.format(contract_text=contract_text)},
        ]
        executed_tool_calls = set()
        history_queries: List[str] = []
        consecutive_repeat_count = 0
        consecutive_empty_searches = 0
        last_observation_empty = False
        force_converge_mode = False

        yield _event("start", {
            "task_id": task_label, "message": "开始合同智能审查流程", "max_turns": max_turns
        })

        for turn in range(max_turns):
            est_prompt_tokens = estimate_prompt_tokens(messages)
            if est_prompt_tokens >= CONTEXT_THRESHOLD_95 and not force_converge_mode:
                force_converge_mode = True
                reason_type = "输入上下文达到 95% 水位线熔断"
                reason_detail = f"累积输入达 {est_prompt_tokens} Tokens (警戒线: {CONTEXT_THRESHOLD_95})"
                print(f"\n[⚡ 熔断告警][{task_label}] 触发 95% 水位线熔断！")
                yield _event("circuit_break", {
                    "task_id": task_label, "reason_type": reason_type,
                    "detail": reason_detail,
                    "recommendations": generate_kb_deficit_recommendation(history_queries, contract_text),
                    "attempted_queries": history_queries,
                })
                messages.append({"role": "user", "content": (
                    "【系统强制干预】：当前检索已被限制，严禁继续发起工具查询！请直接以 Final: 开头输出最终审查报告。"
                )})

            is_last_turn = (turn == max_turns - 1) or force_converge_mode
            yield _event("turn_start", {
                "task_id": task_label, "turn": turn + 1, "max_turns": max_turns,
                "stage": "final_summary" if is_last_turn else "reasoning_and_acting",
                "force_knowledge_mode": force_converge_mode,
            })
            if is_last_turn and not force_converge_mode:
                messages.append({"role": "user", "content": FORCE_FINAL_CONVERGENCE_PROMPT})

            input_chars = sum(len(str(message.get("content", ""))) for message in messages)
            input_tokens = estimate_prompt_tokens(messages)
            print(f"\n{'='*60}\n[*] >>> [{task_label}] 轮次 [{turn+1}/{max_turns}] 发起推理 <<<\n{'='*60}")
            print(
                f"[*] 模型输入: {input_chars} 字符, 估算 {input_tokens} tokens",
                flush=True,
            )
            final_finish_reason = None
            generated_token_count = 0
            usage_prompt_tokens = None
            usage_completion_tokens = None
            reply_chunks = []
            repeated_action_detected = False
            current_max_tokens = OUTPUT_MAX_TOKENS if is_last_turn else REASONING_MAX_TOKENS
            if (
                context_budget_exceeded(input_tokens, current_max_tokens)
                and not force_converge_mode
            ):
                force_converge_mode = True
                is_last_turn = True
                current_max_tokens = OUTPUT_MAX_TOKENS
                messages.append({"role": "user", "content": FORCE_FINAL_CONVERGENCE_PROMPT})
                print(
                    f"[⚠️ 上下文预算熔断][{task_label}] 输入 {input_tokens} tokens + "
                    f"输出预算 {REASONING_MAX_TOKENS} tokens 已达到安全上限 "
                    f"{CONTEXT_THRESHOLD_95} tokens，切换最终报告模式",
                    flush=True,
                )
                yield _event("circuit_break", {
                    "task_id": task_label,
                    "reason_type": "输入与输出预算合计达到上下文安全上限",
                    "detail": (
                        f"输入约 {input_tokens} tokens，输出预算 {REASONING_MAX_TOKENS} tokens，"
                        f"安全上限 {CONTEXT_THRESHOLD_95} tokens"
                    ),
                    "recommendations": generate_kb_deficit_recommendation(
                        history_queries, contract_text
                    ),
                    "attempted_queries": history_queries,
                })
            if debug:
                print(
                    f"\n[DEBUG][{task_label}][轮次 {turn + 1}] "
                    f"发送给大模型的 messages:\n"
                    f"{json.dumps(messages, ensure_ascii=False, indent=2)}",
                    flush=True,
                )
                print(
                    f"[DEBUG][{task_label}][轮次 {turn + 1}] "
                    f"请求参数: model={self.agent.model_name}, "
                    f"temperature={AGENT_CONFIG.get('temperature', 0.0)}, "
                    f"top_p={AGENT_CONFIG.get('top_p', 1.0)}, "
                    f"max_tokens={current_max_tokens}, stream=True",
                    flush=True,
                )
            try:
                stream_response = self.agent.client.chat.completions.create(
                    model=self.agent.model_name, messages=messages,
                    temperature=AGENT_CONFIG.get("temperature", 0.0),
                    top_p=AGENT_CONFIG.get("top_p", 1.0), seed=AGENT_CONFIG.get("seed", 42),
                    max_tokens=current_max_tokens,
                    stop=None if is_last_turn else AGENT_CONFIG.get("stop", ["Observation:"]),
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for chunk in stream_response:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        usage_prompt_tokens = getattr(usage, "prompt_tokens", None)
                        usage_completion_tokens = getattr(usage, "completion_tokens", None)
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        final_finish_reason = choice.finish_reason
                    delta = getattr(choice.delta, "content", None)
                    if delta:
                        reply_chunks.append(delta)
                        generated_token_count += 1
                        yield _event("token", {"token": delta, "task_id": task_label})
                        await asyncio.sleep(0.001)
                        partial_reply = "".join(reply_chunks)
                        actions = extract_actions(partial_reply)
                        if len(actions) >= 2:
                            action_signatures = [
                                f"{name}:{argument.strip()}" for name, argument in actions
                            ]
                            repeated_action_detected = (
                                len(set(action_signatures)) < len(action_signatures)
                                or len(actions) >= 3
                            )
                            if repeated_action_detected:
                                break
            except Exception as exc:
                yield _event("error", {"error": f"底层推理交互异常: {str(exc)}", "task_id": task_label})
                return

            reply = "".join(reply_chunks).strip()
            if debug:
                print(
                    f"\n[DEBUG][{task_label}][轮次 {turn + 1}] "
                    f"大模型完整响应:\n{reply}\n"
                    f"[DEBUG][{task_label}][轮次 {turn + 1}] 响应结束",
                    flush=True,
                )
            output_chars = len(reply)
            output_tokens = usage_completion_tokens or estimate_prompt_tokens([{"content": reply}])
            measured_input_tokens = usage_prompt_tokens or input_tokens
            estimated_total_tokens = measured_input_tokens + output_tokens
            token_source = "model_usage" if usage_completion_tokens is not None else "local_estimate"
            print(
                f"[*] 模型输出: {output_chars} 字符, 估算 {output_tokens} tokens, "
                f"流式片段 {generated_token_count}, 预算 {current_max_tokens} tokens, "
                f"估算总量 {estimated_total_tokens} tokens, "
                f"token_source={token_source}, finish_reason={final_finish_reason}",
                flush=True,
            )
            is_truncated = final_finish_reason == "length"
            yield _event("generation_meta", {
                "task_id": task_label, "turn": turn + 1, "finish_reason": final_finish_reason,
                "token_budget": current_max_tokens, "generated_tokens": output_tokens,
                "stream_chunks": generated_token_count,
                "prompt_chars": input_chars, "est_prompt_tokens": measured_input_tokens,
                "output_chars": output_chars, "output_est_tokens": output_tokens,
                "estimated_total_tokens": estimated_total_tokens,
                "token_source": token_source,
                "is_truncated": is_truncated,
                "used_self_knowledge": force_converge_mode,
            })
            if is_truncated:
                print(
                    f"[⚠️ 输出截断][{task_label}] 第 {turn + 1} 轮达到 "
                    f"{current_max_tokens} tokens 输出上限",
                    flush=True,
                )
                print(
                    f"[⚠️ 输出截断正文开始][{task_label}][轮次 {turn + 1}]\n"
                    f"{reply}\n"
                    f"[⚠️ 输出截断正文结束][{task_label}][轮次 {turn + 1}]",
                    flush=True,
                )
                yield _event("truncation_alert", {
                    "task_id": task_label,
                    "warning": "模型审查意见生成未完成，已触碰最大 Token 上限发生截断！",
                    "turn": turn + 1, "finish_reason": final_finish_reason,
                    "partial_tail": reply[-80:] if len(reply) >= 80 else reply,
                })

            actions = extract_actions(reply)
            if repeated_action_detected:
                force_converge_mode = True
                print(
                    f"[⚠️ 单轮工具调用熔断][{task_label}] "
                    f"检测到 {len(actions)} 个 Action，已提前停止本轮生成",
                    flush=True,
                )
                yield _event("circuit_break", {
                    "task_id": task_label,
                    "reason_type": "单次模型响应连续生成多个工具调用",
                    "detail": f"本轮检测到 {len(actions)} 个 Action，已提前中断生成",
                    "recommendations": generate_kb_deficit_recommendation(history_queries, contract_text),
                    "attempted_queries": history_queries,
                })
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "【系统介入】：检测到单次响应包含多个工具调用，已停止工具检索。请直接以 Final: 开头输出最终审查报告。"},
                ])
                continue

            tool_name, tool_arg = actions[0] if actions else (None, None)
            if tool_name and is_truncated and not is_last_turn:
                force_converge_mode = True
                print(
                    f"[⚠️ 工具调用截断][{task_label}] 第 {turn + 1} 轮未完整结束，"
                    "停止检索并转入最终报告",
                    flush=True,
                )
                yield _event("circuit_break", {
                    "task_id": task_label,
                    "reason_type": "工具调用响应达到输出上限",
                    "detail": f"本轮达到 {current_max_tokens} tokens 输出上限",
                    "recommendations": generate_kb_deficit_recommendation(history_queries, contract_text),
                    "attempted_queries": history_queries,
                })
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "【系统介入】：工具调用响应被截断，已停止外部检索。请直接以 Final: 开头输出最终审查报告。"},
                ])
                continue
            if tool_name and force_converge_mode:
                if is_last_turn:
                    yield _event("final_report", {
                        "task_id": task_label,
                        "raw_report": clean_report_content(reply),
                        "turns": turn + 1,
                        "is_complete": False,
                        "finish_reason": final_finish_reason,
                        "status": "forced_convergence_failed",
                        "self_knowledge_mode": True,
                    })
                    yield _event("done", {
                        "task_id": task_label,
                        "message": "审查流程结束（模型未按熔断要求输出 Final）",
                    })
                    return
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "【系统强制干预】：当前检索已被限制，严禁继续发起工具查询！请直接以 Final: 开头输出最终审查报告。"},
                ])
                continue

            if tool_name and is_last_turn:
                yield _event("final_report", {
                    "task_id": task_label,
                    "raw_report": clean_report_content(reply),
                    "turns": turn + 1,
                    "is_complete": False,
                    "finish_reason": final_finish_reason,
                    "status": "tool_call_on_final_turn",
                    "self_knowledge_mode": False,
                })
                yield _event("done", {"task_id": task_label, "message": "审查流程结束（达到最大轮次）"})
                return

            if tool_name and tool_name in self.agent.tool_mapping and not is_truncated:
                action_sig = f"{tool_name}:{str(tool_arg).strip() if tool_arg else ''}"
                history_queries.append(str(tool_arg).strip())
                if action_sig in executed_tool_calls:
                    consecutive_repeat_count += 1
                else:
                    consecutive_repeat_count = 0
                    executed_tool_calls.add(action_sig)
                is_repeat = consecutive_repeat_count >= 1
                is_budget_exceeded = len(executed_tool_calls) >= max_search_budget
                is_empty_loop = consecutive_empty_searches >= 2
                if is_repeat or is_budget_exceeded or is_empty_loop:
                    if is_empty_loop or (is_repeat and last_observation_empty):
                        reason_type, detail = "知识库检索空召回（命中知识库盲区）", f"连续 {consecutive_empty_searches} 次未在知识库中匹配到法条"
                    elif is_repeat:
                        reason_type, detail = "重复调用同一指令（死循环死锁）", f"连续多次执行相同动作 `{action_sig}`"
                    else:
                        reason_type, detail = "检索轮次预算耗尽", f"已执行检索次数达到安全预算 ({len(executed_tool_calls)}/{max_search_budget})"
                    print(f"\n[⚠️ 动态熔断拦截生效][{task_label}] 原因: {reason_type}")
                    yield _event("circuit_break", {
                        "task_id": task_label, "reason_type": reason_type, "detail": detail,
                        "recommendations": generate_kb_deficit_recommendation(history_queries, contract_text),
                        "attempted_queries": history_queries,
                    })
                    force_converge_mode = True
                    messages.extend([
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": f"【系统介入】：检测到检索陷入瓶颈（{reason_type}）。请立即停止外部查询，直接基于大模型既有法律常识，在下一轮直接以 Final: 给出终审结论。"},
                    ])
                    continue

                yield _event("tool_start", {"task_id": task_label, "tool": tool_name, "query": tool_arg})
                try:
                    observation = self.agent.tool_mapping[tool_name](tool_arg)
                except Exception as exc:
                    observation = f"工具执行异常: {str(exc)}"
                if debug:
                    print(
                        f"\n[DEBUG][{task_label}][轮次 {turn + 1}] 工具调用: "
                        f"{tool_name}({tool_arg})\n"
                        f"[DEBUG][{task_label}][轮次 {turn + 1}] 工具返回:\n"
                        f"{observation}",
                        flush=True,
                    )
                last_observation_empty = (
                    not observation.strip()
                    or "未检索到" in observation
                    or "未找到任何" in observation
                    or "知识库为空" in observation
                )
                consecutive_empty_searches = consecutive_empty_searches + 1 if last_observation_empty else 0
                yield _event("tool_result", {"task_id": task_label, "tool": tool_name, "observation": observation})
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": format_observation(observation)},
                ])
                if last_observation_empty:
                    messages.append({
                        "role": "user",
                        "content": EMPTY_SEARCH_RETRY_PROMPT,
                    })
                continue

            if is_final_report(reply) or is_last_turn or is_truncated:
                yield _event("final_report", {
                    "task_id": task_label, "raw_report": clean_report_content(reply),
                    "turns": turn + 1, "is_complete": not is_truncated,
                    "finish_reason": final_finish_reason,
                    "status": "truncated" if is_truncated else "success",
                    "self_knowledge_mode": force_converge_mode,
                })
                yield _event("done", {"task_id": task_label, "message": "审查流程结束"})
                return

            messages.extend([
                {"role": "assistant", "content": reply},
                {"role": "user", "content": TOOL_CALL_RETRY_PROMPT},
            ])

        yield _event("timeout", {"task_id": task_label, "message": "已达到最大审查轮数，自动终止"})
