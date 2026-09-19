#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: gateway/api_server.py
职责:
1. 暴露 RESTful 接口供 Web 前端调用
2. 提供合同文件（Word / PDF / 图片 OCR / TXT）上传与条款切分接口
3. 提供基于 SSE (Server-Sent Events) 的打字机式流式推理审查接口
4. 核心健壮性与可观测性升级：
   - 实时监控：打印每轮推理输入/输出的字符数与估算 Token 开销
   - 物理截断告警：捕获 finish_reason == 'length' 并透出 truncation_alert
   - 全程充裕预算：将生成上限锁定为 4096 Tokens，杜绝长文审查截断
   - 终审防误杀：智能识别漏写 Final: 标签的审查报告，并在末轮剥离 stop 词
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

import re
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# 项目根目录对齐
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.config import GATEWAY_HOST, GATEWAY_PORT, DATA_DIR, AGENT_CONFIG
from core.schemas import ReviewRequest, AgentExecutionResult, ContractReviewReport
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

# 初始化 FastAPI 应用
app = FastAPI(
    title="Legal Agent Lab API Gateway",
    description="本地法务合同审查智能体高并发网关服务",
    version="1.4.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件单例
agent_instance = ContractReviewAgent()
doc_loader = DocumentLoader()


def is_final_report(text: str) -> bool:
    """
    智能判定模型当前是否已进入终审报告阶段。
    解决模型输出漏写 'Final:' 前缀时被误判为违规格式的痛点。
    """
    if not text:
        return False

    # 1. 显式命中标识
    if any(k in text for k in ["Final:", "【最终结论】", "最终审查意见", "综合审查报告"]):
        return True

    # 2. 语义特征识别：包含 2 个以上法务结构化标签，认定为报告
    report_signatures = [
        "风险等级",
        "修改建议",
        "法律依据",
        "合规依据",
        "审查结论",
        "### 1.",
        "1. 违约",
        "1. 争议",
    ]
    matches = sum(1 for sig in report_signatures if sig in text)
    return matches >= 2


def clean_report_content(raw_text: str) -> str:
    """提取纯净的审查报告正文，剔除前导标识"""
    prefixes = ["Final:", "【最终结论】:", "【最终结论】", "最终审查意见:"]
    cleaned = raw_text.strip()
    for p in prefixes:
        if p in cleaned:
            cleaned = cleaned.split(p, 1)[-1].strip()
            break
    return cleaned


# ==================== 1. 基础连通性与健康检查 ====================

@app.get("/health")
async def health_check():
    """网关与推理服务存活检测"""
    return {
        "status": "healthy",
        "model": agent_instance.model_name,
        "tools_ready": list(agent_instance.tool_mapping.keys()),
    }


# ==================== 2. 会话生命周期接口 ====================

@app.post("/api/v1/sessions/create")
async def create_session():
    """创建一个独立的合同审查会话"""
    session_id = session_manager.create_session()
    return {"code": 200, "message": "success", "data": {"session_id": session_id}}


# ==================== 3. 合同上传与条款结构化切分 ====================

@app.post("/api/v1/contract/upload")
async def upload_contract(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    接收上传的 Word (.docx)、PDF (.pdf)、图片或 TXT，执行 OCR 与条款切分
    """
    if not session_id or not session_manager.get_session(session_id):
        session_id = session_manager.create_session()

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = upload_dir / f"{session_id}_{file.filename}"

    try:
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # 解析并执行条款切分
        clauses = doc_loader.load_and_split(temp_file_path)
        clauses_data = [c.to_dict() for c in clauses]

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


# ==================== 4. 阻塞式审查接口 ====================

@app.post("/api/v1/contract/review/sync", response_model=AgentExecutionResult)
async def review_contract_sync(request: ReviewRequest):
    """
    阻塞式全量审查接口：等待 Agent 完成 ReAct 循环后返回完整报告
    """
    result = agent_instance.run_contract_agent(
        contract_text=request.contract_text,
        max_turns=request.max_turns or AGENT_CONFIG.get("max_turns", 6),
    )
    return result


# ==================== 5. 核心：带长度遥测与截断告警的 SSE 流式审查接口 ====================

@app.post("/api/v1/contract/review/stream")
async def review_contract_stream(request: ReviewRequest):
    """
    基于 Server-Sent Events (SSE) 的实时流式审查接口
    """
    limit_turns = request.max_turns or AGENT_CONFIG.get("max_turns", 6)

    async def event_generator():
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTRACT_INPUT_TEMPLATE.format(contract_text=request.contract_text)},
        ]

        yield {
            "event": "start",
            "data": json.dumps({"message": "开始合同智能审查流程", "max_turns": limit_turns}, ensure_ascii=False),
        }

        for turn in range(limit_turns):
            is_last_turn = (turn == limit_turns - 1)

            yield {
                "event": "turn_start",
                "data": json.dumps({
                    "turn": turn + 1,
                    "max_turns": limit_turns,
                    "stage": "final_summary" if is_last_turn else "reasoning_and_acting"
                }, ensure_ascii=False),
            }

            if is_last_turn:
                messages.append({"role": "user", "content": FORCE_FINAL_CONVERGENCE_PROMPT})

            # ---------------- 输入长度与 Token 统计追踪 ----------------
            total_prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
            est_prompt_tokens = int(total_prompt_chars * 1.3)

            print(f"\n{'='*55}")
            print(f"[*] >>> 轮次 [{turn+1}/{limit_turns}] 发起推理 <<<")
            print(f"[*] 历史消息条数: {len(messages)} 条")
            print(f"[*] 本轮输入总长度: {total_prompt_chars} 字符 (估算 ~{est_prompt_tokens} Tokens)")
            print(f"{'='*55}")

            # 统一给足 4096，防止任何一轮输出长报告时发生物理截断
            current_max_tokens = 4096
            # 终审轮次剥离 stop 词，防止报告正文因匹配而异常中断
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
                            "data": json.dumps({"token": delta}, ensure_ascii=False),
                        }
                        await asyncio.sleep(0.001)

            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": f"底层推理交互异常: {str(e)}"}, ensure_ascii=False),
                }
                return

            reply = "".join(reply_chunks).strip()

            # ---------------- 输出长度与物理截断统计 ----------------
            output_chars = len(reply)
            is_truncated = (final_finish_reason == "length")

            print(f"{'-'*55}")
            print(f"[*] <<< 轮次 [{turn+1}/{limit_turns}] 推理结束 <<<")
            print(f"[*] 本轮实际生成 Token: {generated_token_count} (预算上限: {current_max_tokens})")
            print(f"[*] 本轮输出正文字符数: {output_chars} 字符")
            print(f"[*] 结束标识 (finish_reason): {final_finish_reason}")
            if is_truncated:
                print(f"[⚠️ 截断告警] 轮次 {turn+1} 输出被物理掐断！已耗尽 {current_max_tokens} Tokens 上限。")
            print(f"{'-'*55}\n")

            # 将统计与元数据透出给前端
            yield {
                "event": "generation_meta",
                "data": json.dumps({
                    "turn": turn + 1,
                    "finish_reason": final_finish_reason,
                    "token_budget": current_max_tokens,
                    "generated_tokens": generated_token_count,
                    "prompt_chars": total_prompt_chars,
                    "est_prompt_tokens": est_prompt_tokens,
                    "output_chars": output_chars,
                    "is_truncated": is_truncated,
                }, ensure_ascii=False),
            }

            if is_truncated:
                yield {
                    "event": "truncation_alert",
                    "data": json.dumps({
                        "warning": "模型审查意见生成未完成，已触碰最大 Token 上限发生截断！",
                        "turn": turn + 1,
                        "finish_reason": final_finish_reason,
                        "partial_tail": reply[-80:] if len(reply) >= 80 else reply,
                    }, ensure_ascii=False),
                }

            # ---------------- 分支 1: 执行工具调用 ----------------
            tool_name, tool_arg = extract_action(reply)
            if tool_name and tool_name in agent_instance.tool_mapping and not is_last_turn and not is_truncated:
                yield {
                    "event": "tool_start",
                    "data": json.dumps({"tool": tool_name, "query": tool_arg}, ensure_ascii=False),
                }

                try:
                    observation = agent_instance.tool_mapping[tool_name](tool_arg)
                except Exception as ex:
                    observation = f"工具执行异常: {str(ex)}"

                yield {
                    "event": "tool_result",
                    "data": json.dumps({"tool": tool_name, "observation": observation}, ensure_ascii=False),
                }

                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": format_observation(observation)})
                continue

            # ---------------- 分支 2: 终审报告收敛判定 ----------------
            if is_final_report(reply) or is_last_turn or is_truncated:
                cleaned_report = clean_report_content(reply)
                parsed_report = None

                # 兼容性解析 JSON
                try:
                    clean_str = cleaned_report
                    if "```" in clean_str:
                        clean_str = re.sub(r"^```(?:json)?\s*", "", clean_str, flags=re.MULTILINE)
                        clean_str = re.sub(r"\s*```$", "", clean_str, flags=re.MULTILINE).strip()

                    match = re.search(r"(\{[\s\S]*\})", clean_str)
                    if match:
                        parsed_dict = json.loads(match.group(1))
                        parsed_report = ContractReviewReport.model_validate(parsed_dict).model_dump()
                except Exception:
                    parsed_report = None

                yield {
                    "event": "final_report",
                    "data": json.dumps({
                        "report": parsed_report,
                        "raw_report": cleaned_report,
                        "turns": turn + 1,
                        "is_complete": not is_truncated,
                        "finish_reason": final_finish_reason,
                        "status": "truncated" if is_truncated else "success",
                    }, ensure_ascii=False),
                }
                yield {
                    "event": "done",
                    "data": json.dumps({"message": "审查流程结束", "is_complete": not is_truncated}, ensure_ascii=False),
                }
                return

            # ---------------- 分支 3: 格式异常重试补偿 ----------------
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": TOOL_CALL_RETRY_PROMPT})

        yield {
            "event": "timeout",
            "data": json.dumps({"message": "已达到最大审查轮数，自动终止", "is_complete": False}, ensure_ascii=False),
        }

    return EventSourceResponse(event_generator())


# ==================== 启动主入口 ====================
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