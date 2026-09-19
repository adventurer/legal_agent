"""Streaming contract-review orchestration for the gateway."""

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List

from configs.config import AGENT_CONFIG
from core.agent_loop import extract_action
from core.prompts import (
    AGENT_SYSTEM_PROMPT,
    FORCE_FINAL_CONVERGENCE_PROMPT,
    TOOL_CALL_RETRY_PROMPT,
    USER_CONTRACT_INPUT_TEMPLATE,
    format_observation,
)
from services.report_parser import clean_report_content, is_final_report

MODEL_MAX_CONTEXT = AGENT_CONFIG.get("max_context_tokens", 8192)
CONTEXT_THRESHOLD_95 = int(MODEL_MAX_CONTEXT * 0.95)
OUTPUT_MAX_TOKENS = 4096


def generate_kb_deficit_recommendation(
    attempted_queries: List[str], text_sample: str
) -> List[str]:
    """Generate concrete legal sources suggested by failed or exhausted searches."""
    recommendations = []
    combined_context = (" ".join(attempted_queries) + " " + text_sample).lower()

    if any(k in combined_context for k in ["仲裁", "独任", "仲裁员", "仲裁委员会"]):
        recommendations.append("《中华人民共和国仲裁法》（重点补齐：仲裁协议有效要件、仲裁委员会选定明确性、独任仲裁员选任程序）")
    if any(k in combined_context for k in ["诉讼", "管辖", "管辖权", "法院"]):
        recommendations.append("《中华人民共和国民事诉讼法》（重点补齐：协议管辖法定连接点、专属管辖及级别管辖限制）")
    if any(k in combined_context for k in ["知识产权", "软件著作权", "专利", "商业秘密"]):
        recommendations.append("《中华人民共和国著作权法》及《反不正当竞争法》（重点补齐：职务成果权属约定、商业秘密保护边界）")
    if any(k in combined_context for k in ["税", "发票", "代缴"]):
        recommendations.append("国家税务总局《发票管理办法》及增值税专用发票开具与货款结算联动规则")
    if any(k in combined_context for k in ["劳动", "竞业限制", "辞职", "社保"]):
        recommendations.append("《中华人民共和国劳动合同法》（重点补齐：竞业限制经济补偿标准、用人单位单方解除权）")

    if not recommendations:
        clean_terms = [q for q in attempted_queries if q.strip()]
        terms_str = "、".join(f"'{t}'" for t in clean_terms[:3]) if clean_terms else "相关专业领域法规"
        recommendations.append(f"针对关键词 [{terms_str}] 的专项部委规章、司法解释或企业合规制度细则")
    return recommendations


def _event(name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


class ReviewOrchestrator:
    """Owns the SSE ReAct loop while leaving HTTP concerns in api_server."""

    def __init__(self, agent: Any):
        self.agent = agent

    async def stream(
        self, contract_text: str, max_turns: int, task_label: str
    ) -> AsyncGenerator[Dict[str, str], None]:
        max_search_budget = max(3, max_turns - 1)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTRACT_INPUT_TEMPLATE.format(contract_text=contract_text)},
        ]
        executed_tool_calls = set()
        history_queries: List[str] = []
        consecutive_repeat_count = 0
        consecutive_empty_searches = 0
        force_converge_mode = False

        yield _event("start", {
            "task_id": task_label, "message": "开始合同智能审查流程", "max_turns": max_turns
        })

        for turn in range(max_turns):
            total_prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            est_prompt_tokens = int(total_prompt_chars * 1.3)
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
                    "【系统警告：输入上下文已达到 95% 水位线】：上下文空间即将耗尽，系统已关闭所有外部工具检索权限！"
                    "请立即停止任何工具调用，直接依据大模型已掌握的法律法学原理与通用常识完成审查推理，"
                    "并在本次回答中直接以 Final: 开头输出完整的最终审查报告。"
                )})

            is_last_turn = (turn == max_turns - 1) or force_converge_mode
            yield _event("turn_start", {
                "task_id": task_label, "turn": turn + 1, "max_turns": max_turns,
                "stage": "final_summary" if is_last_turn else "reasoning_and_acting",
                "force_knowledge_mode": force_converge_mode,
            })
            if is_last_turn and not force_converge_mode:
                messages.append({"role": "user", "content": FORCE_FINAL_CONVERGENCE_PROMPT})

            print(f"\n{'='*60}\n[*] >>> [{task_label}] 轮次 [{turn+1}/{max_turns}] 发起推理 <<<\n{'='*60}")
            final_finish_reason = None
            generated_token_count = 0
            reply_chunks = []
            try:
                stream_response = self.agent.client.chat.completions.create(
                    model=self.agent.model_name, messages=messages,
                    temperature=AGENT_CONFIG.get("temperature", 0.0),
                    top_p=AGENT_CONFIG.get("top_p", 1.0), seed=AGENT_CONFIG.get("seed", 42),
                    max_tokens=OUTPUT_MAX_TOKENS,
                    stop=None if is_last_turn else AGENT_CONFIG.get("stop", ["Observation:"]),
                    stream=True,
                )
                for chunk in stream_response:
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
            except Exception as exc:
                yield _event("error", {"error": f"底层推理交互异常: {str(exc)}", "task_id": task_label})
                return

            reply = "".join(reply_chunks).strip()
            is_truncated = final_finish_reason == "length"
            yield _event("generation_meta", {
                "task_id": task_label, "turn": turn + 1, "finish_reason": final_finish_reason,
                "token_budget": OUTPUT_MAX_TOKENS, "generated_tokens": generated_token_count,
                "prompt_chars": total_prompt_chars, "est_prompt_tokens": est_prompt_tokens,
                "output_chars": len(reply), "is_truncated": is_truncated,
                "used_self_knowledge": force_converge_mode,
            })
            if is_truncated:
                yield _event("truncation_alert", {
                    "task_id": task_label,
                    "warning": "模型审查意见生成未完成，已触碰最大 Token 上限发生截断！",
                    "turn": turn + 1, "finish_reason": final_finish_reason,
                    "partial_tail": reply[-80:] if len(reply) >= 80 else reply,
                })

            tool_name, tool_arg = extract_action(reply)
            if tool_name and (force_converge_mode or is_last_turn):
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "【系统强制干预】：当前检索已被限制，严禁继续发起工具查询！请直接以 Final: 开头输出最终审查报告。"},
                ])
                continue

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
                    if is_repeat:
                        reason_type, detail = "重复调用同一指令（死循环死锁）", f"连续多次执行相同动作 `{action_sig}`"
                    elif is_empty_loop:
                        reason_type, detail = "知识库检索空召回（命中知识库盲区）", f"连续 {consecutive_empty_searches} 次未在知识库中匹配到法条"
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
                consecutive_empty_searches = consecutive_empty_searches + 1 if "未检索到" in observation or not observation.strip() else 0
                yield _event("tool_result", {"task_id": task_label, "tool": tool_name, "observation": observation})
                messages.extend([
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": format_observation(observation)},
                ])
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
