#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/api_server.py
职责:
1. 暴露 RESTful 接口供 Web 前端调用
2. 提供合同文件（Word / PDF / 图片 OCR / TXT）上传与条款切分接口
3. 提供基于 SSE 的打字机式流式推理审查接口
4. 核心健壮性与可观测性：
   - 任务级端到端追踪：日志全程注入 Task ID 前缀
   - 熔断事件透传：新增 circuit_break SSE 事件，实时向前端推送熔断详情与知识库补齐建议
   - 纯输入 Token 95% 熔断底线：输入接近极限时自动切入模型自知识推理
   - 全程 4096 预算，末轮解除 stop 词杜绝长文截断
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

import re
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# 项目根目录对齐
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.config import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    DATA_DIR,
    AGENT_CONFIG,
    MAX_UPLOAD_BYTES,
)
from core.schemas import ReviewRequest, AgentExecutionResult
from core.prompts import (
    AGENT_SYSTEM_PROMPT,
    USER_CONTRACT_INPUT_TEMPLATE,
    FORCE_FINAL_CONVERGENCE_PROMPT,
    TOOL_CALL_RETRY_PROMPT,
    format_observation,
)
from core.agent_loop import ContractReviewAgent, extract_action
from services.doc_loader import DocumentLoader
from gateway.session_manager import session_manager
from gateway.upload_service import create_temp_upload_path, get_safe_extension, save_upload_file
from services.report_parser import clean_report_content, is_final_report

# 模型上下文窗口规格定义（默认参考 8192 窗口）
MODEL_MAX_CONTEXT = AGENT_CONFIG.get("max_context_tokens", 8192)
CONTEXT_THRESHOLD_95 = int(MODEL_MAX_CONTEXT * 0.95)  # 95% 输入水位线
OUTPUT_MAX_TOKENS = 4096


app = FastAPI(
    title="Legal Agent Lab API Gateway",
    description="本地法务合同审查智能体高并发网关服务",
    version="1.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance = ContractReviewAgent()
doc_loader = DocumentLoader()


def normalize_clauses_output(raw_clauses: Any) -> List[Dict[str, Any]]:
    formatted = []
    if not raw_clauses:
        return formatted
    for i, c in enumerate(raw_clauses):
        if hasattr(c, "to_dict") and callable(c.to_dict):
            formatted.append(c.to_dict())
        elif isinstance(c, dict):
            formatted.append({
                "index": c.get("index", i + 1),
                "title": c.get("title", f"条款 {i + 1}"),
                "content": c.get("content", ""),
                "type": c.get("type", "article"),
            })
        else:
            formatted.append({
                "index": getattr(c, "index", i + 1),
                "title": getattr(c, "title", f"条款 {i + 1}"),
                "content": getattr(c, "content", str(c)),
                "type": getattr(c, "type", "article"),
            })
    return formatted


def generate_kb_deficit_recommendation(attempted_queries: List[str], text_sample: str) -> List[str]:
    """生成需要向知识库增补的具体法规清单"""
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
        terms_str = "、".join([f"'{t}'" for t in clean_terms[:3]]) if clean_terms else "相关专业领域法规"
        recommendations.append(f"针对关键词 [{terms_str}] 的专项部委规章、司法解释或企业合规制度细则")

    return recommendations


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": agent_instance.model_name,
        "tools_ready": list(agent_instance.tool_mapping.keys()),
        "context_threshold_95": CONTEXT_THRESHOLD_95,
    }


@app.post("/api/v1/sessions/create")
async def create_session():
    session_id = session_manager.create_session()
    return {"code": 200, "message": "success", "data": {"session_id": session_id}}


@app.post("/api/v1/contract/upload")
async def upload_contract(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    if not session_id or not session_manager.get_session(session_id):
        session_id = session_manager.create_session()

    extension = get_safe_extension(file.filename or "")
    upload_dir = DATA_DIR / "uploads"
    temp_file_path = create_temp_upload_path(upload_dir, session_id, extension)

    try:
        await save_upload_file(file, temp_file_path, MAX_UPLOAD_BYTES)

        raw_clauses = doc_loader.load_and_split(temp_file_path)
        clauses_data = normalize_clauses_output(raw_clauses)
        session_manager.store_clauses(session_id, clauses_data)

        return {
            "code": 200,
            "message": "success",
            "data": {
                "session_id": session_id,
                "filename": file.filename,
                "clause_count": len(clauses_data),
                "clauses": clauses_data,
            },
        }
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError, ImportError) as e:
        raise HTTPException(status_code=422, detail=f"文件处理失败: {e}") from e
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@app.post("/api/v1/contract/review/stream")
async def review_contract_stream(request: ReviewRequest):
    limit_turns = request.max_turns or AGENT_CONFIG.get("max_turns", 6)
    max_search_budget = max(3, limit_turns - 1)

    clause_title_match = re.search(r"^(第[一二三四五六七八九十百0-9]+条[^\n]+|合同前言[^\n]*)", request.contract_text.strip())
    task_label = clause_title_match.group(1).strip() if clause_title_match else f"Task-{uuid.uuid4().hex[:6]}"

    async def event_generator():
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTRACT_INPUT_TEMPLATE.format(contract_text=request.contract_text)},
        ]

        executed_tool_calls = set()
        history_queries: List[str] = []
        consecutive_repeat_count = 0
        consecutive_empty_searches = 0
        force_converge_mode = False
        break_reason_type = ""
        break_reason_detail = ""

        yield {
            "event": "start",
            "data": json.dumps({"task_id": task_label, "message": "开始合同智能审查流程", "max_turns": limit_turns}, ensure_ascii=False),
        }

        for turn in range(limit_turns):
            # 1. 95% 输入 Token 前置检测
            total_prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            est_prompt_tokens = int(total_prompt_chars * 1.3)
            is_prompt_over_95 = (est_prompt_tokens >= CONTEXT_THRESHOLD_95)

            if is_prompt_over_95 and not force_converge_mode:
                force_converge_mode = True
                break_reason_type = "输入上下文达到 95% 水位线熔断"
                break_reason_detail = f"累积输入达 {est_prompt_tokens} Tokens (警戒线: {CONTEXT_THRESHOLD_95})"
                recs = generate_kb_deficit_recommendation(history_queries, request.contract_text)
                print(f"\n[⚡ 熔断告警][{task_label}] 触发 95% 水位线熔断！")

                # 向前端推送熔断与知识库增补事件
                yield {
                    "event": "circuit_break",
                    "data": json.dumps({
                        "task_id": task_label,
                        "reason_type": break_reason_type,
                        "detail": break_reason_detail,
                        "recommendations": recs,
                        "attempted_queries": history_queries,
                    }, ensure_ascii=False),
                }

                messages.append({
                    "role": "user",
                    "content": (
                        "【系统警告：输入上下文已达到 95% 水位线】：上下文空间即将耗尽，系统已关闭所有外部工具检索权限！"
                        "请立即停止任何工具调用，直接依据大模型已掌握的法律法学原理与通用常识完成审查推理，"
                        "并在本次回答中直接以 Final: 开头输出完整的最终审查报告。"
                    )
                })

            is_last_turn = (turn == limit_turns - 1) or force_converge_mode

            yield {
                "event": "turn_start",
                "data": json.dumps({
                    "task_id": task_label,
                    "turn": turn + 1,
                    "max_turns": limit_turns,
                    "stage": "final_summary" if is_last_turn else "reasoning_and_acting",
                    "force_knowledge_mode": force_converge_mode,
                }, ensure_ascii=False),
            }

            if is_last_turn and not force_converge_mode:
                messages.append({"role": "user", "content": FORCE_FINAL_CONVERGENCE_PROMPT})

            print(f"\n{'='*60}\n[*] >>> [{task_label}] 轮次 [{turn+1}/{limit_turns}] 发起推理 <<<\n{'='*60}")

            current_max_tokens = OUTPUT_MAX_TOKENS
            current_stops = None if is_last_turn else AGENT_CONFIG.get("stop", ["Observation:"])

            reply_chunks = []
            final_finish_reason = None
            generated_token_count = 0

            try:
                stream_response = agent_instance.client.chat.completions.create(
                    model=agent_instance.model_name,
                    messages=messages,
                    temperature=AGENT_CONFIG.get("temperature", 0.0),
                    top_p=AGENT_CONFIG.get("top_p", 1.0),
                    seed=AGENT_CONFIG.get("seed", 42),
                    max_tokens=current_max_tokens,
                    stop=current_stops,
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
                        yield {
                            "event": "token",
                            "data": json.dumps({"token": delta, "task_id": task_label}, ensure_ascii=False),
                        }
                        await asyncio.sleep(0.001)

            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"底层推理交互异常: {str(e)}", "task_id": task_label}, ensure_ascii=False),
                }
                return

            reply = "".join(reply_chunks).strip()
            output_chars = len(reply)
            is_truncated = (final_finish_reason == "length")

            yield {
                "event": "generation_meta",
                "data": json.dumps({
                    "task_id": task_label,
                    "turn": turn + 1,
                    "finish_reason": final_finish_reason,
                    "token_budget": current_max_tokens,
                    "generated_tokens": generated_token_count,
                    "prompt_chars": total_prompt_chars,
                    "est_prompt_tokens": est_prompt_tokens,
                    "output_chars": output_chars,
                    "is_truncated": is_truncated,
                    "used_self_knowledge": force_converge_mode,
                }, ensure_ascii=False),
            }

            if is_truncated:
                yield {
                    "event": "truncation_alert",
                    "data": json.dumps({
                        "task_id": task_label,
                        "warning": "模型审查意见生成未完成，已触碰最大 Token 上限发生截断！",
                        "turn": turn + 1,
                        "finish_reason": final_finish_reason,
                        "partial_tail": reply[-80:] if len(reply) >= 80 else reply,
                    }, ensure_ascii=False),
                }

            # 工具调用与熔断拦截
            tool_name, tool_arg = extract_action(reply)
            if tool_name and (force_converge_mode or is_last_turn):
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": "【系统强制干预】：当前检索已被限制，严禁继续发起工具查询！请直接以 Final: 开头输出最终审查报告。"
                })
                continue

            if tool_name and tool_name in agent_instance.tool_mapping and not is_truncated:
                current_action_sig = f"{tool_name}:{str(tool_arg).strip() if tool_arg else ''}"
                history_queries.append(str(tool_arg).strip())

                if current_action_sig in executed_tool_calls:
                    consecutive_repeat_count += 1
                else:
                    consecutive_repeat_count = 0
                    executed_tool_calls.add(current_action_sig)

                is_identical_repeat = (consecutive_repeat_count >= 1)
                is_budget_exceeded = (len(executed_tool_calls) >= max_search_budget)
                is_empty_loop = (consecutive_empty_searches >= 2)

                if is_identical_repeat or is_budget_exceeded or is_empty_loop:
                    if is_identical_repeat:
                        break_reason_type = "重复调用同一指令（死循环死锁）"
                        break_reason_detail = f"连续多次执行相同动作 `{current_action_sig}`"
                    elif is_empty_loop:
                        break_reason_type = "知识库检索空召回（命中知识库盲区）"
                        break_reason_detail = f"连续 {consecutive_empty_searches} 次未在知识库中匹配到法条"
                    else:
                        break_reason_type = "检索轮次预算耗尽"
                        break_reason_detail = f"已执行检索次数达到安全预算 ({len(executed_tool_calls)}/{max_search_budget})"

                    recs = generate_kb_deficit_recommendation(history_queries, request.contract_text)
                    print(f"\n[⚠️ 动态熔断拦截生效][{task_label}] 原因: {break_reason_type}")

                    # 推送熔断通知给前端
                    yield {
                        "event": "circuit_break",
                        "data": json.dumps({
                            "task_id": task_label,
                            "reason_type": break_reason_type,
                            "detail": break_reason_detail,
                            "recommendations": recs,
                            "attempted_queries": history_queries,
                        }, ensure_ascii=False),
                    }

                    force_converge_mode = True
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"【系统介入】：检测到检索陷入瓶颈（{break_reason_type}）。请立即停止外部查询，"
                            "直接基于大模型既有法律常识，在下一轮直接以 Final: 给出终审结论。"
                        )
                    })
                    continue

                yield {
                    "event": "tool_start",
                    "data": json.dumps({"task_id": task_label, "tool": tool_name, "query": tool_arg}, ensure_ascii=False),
                }

                try:
                    observation = agent_instance.tool_mapping[tool_name](tool_arg)
                except Exception as ex:
                    observation = f"工具执行异常: {str(ex)}"

                if "未检索到" in observation or not observation.strip():
                    consecutive_empty_searches += 1
                else:
                    consecutive_empty_searches = 0

                yield {
                    "event": "tool_result",
                    "data": json.dumps({"task_id": task_label, "tool": tool_name, "observation": observation}, ensure_ascii=False),
                }

                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": format_observation(observation)})
                continue

            # 终审报告输出判定
            if is_final_report(reply) or is_last_turn or is_truncated:
                cleaned_report = clean_report_content(reply)
                yield {
                    "event": "final_report",
                    "data": json.dumps({
                        "task_id": task_label,
                        "raw_report": cleaned_report,
                        "turns": turn + 1,
                        "is_complete": not is_truncated,
                        "finish_reason": final_finish_reason,
                        "status": "truncated" if is_truncated else "success",
                        "self_knowledge_mode": force_converge_mode,
                    }, ensure_ascii=False),
                }
                yield {
                    "event": "done",
                    "data": json.dumps({"task_id": task_label, "message": "审查流程结束"}, ensure_ascii=False),
                }
                return

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": TOOL_CALL_RETRY_PROMPT})

        yield {
            "event": "timeout",
            "data": json.dumps({"task_id": task_label, "message": "已达到最大审查轮数，自动终止"}, ensure_ascii=False),
        }

    return EventSourceResponse(event_generator())


def main():
    print(f"[*] 正在启动 Legal Agent FastAPI Gateway 监听: http://{GATEWAY_HOST}:{GATEWAY_PORT}")
    uvicorn.run(
        "gateway.api_server:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()