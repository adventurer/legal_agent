#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: core/agent_loop.py
职责:
1. 维护 ReAct 循环 (Thought -> Action -> Observation -> Final)
2. 全面对接 configs/config.py 全局配置
3. 动态调度三大核心工具：
   - search_civil_code: 民法典等法律法规 PDF 依据
   - get_company_policy: 公司合规手册等 PDF 依据
   - get_past_review_rules: 企业自编法典与审查偏好 (SQLite 记忆库)
4. 输出符合 core/schemas.py 契约的强类型结构化报告
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable

import httpx
from openai import OpenAI

# 确保项目根目录在 sys.path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 1. 导入全局配置
from configs.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    DEFAULT_MODEL_NAME,
    REFERENCE_DOCS_DIR,
    RULE_BOOK_DB_PATH,
    AGENT_CONFIG,
    TIMEOUT_CONFIG,
)

# 2. 导入提示词与结构化数据规范
from core.prompts import (
    AGENT_SYSTEM_PROMPT,
    USER_CONTRACT_INPUT_TEMPLATE,
    FORCE_FINAL_CONVERGENCE_PROMPT,
    TOOL_CALL_RETRY_PROMPT,
    format_observation,
)
from core.schemas import AgentExecutionResult
from services.report_parser import split_final_output

# 3. 导入底层知识库与企业自编法典服务
try:
    from services.pdf_kb import PDFKnowledgeBase
except ImportError:
    PDFKnowledgeBase = None

try:
    from services.rule_book import EnterpriseRuleBook
except ImportError:
    EnterpriseRuleBook = None


def extract_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析单行 Action，兼容半角/全角括号和带引号参数。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Action:"):
            content = line[len("Action:"):].strip()
            match = re.fullmatch(r"([A-Za-z_][\w-]*)\s*[\(（](.*)[\)）]", content)
            if match:
                name, arg = match.groups()
                return name.strip(), arg.strip().strip("'\"“”‘’")
    return None, None


class ContractReviewAgent:
    """合同审查智能体调度循环"""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        api_key: str = VLLM_API_KEY,
        model_name: str = DEFAULT_MODEL_NAME,
        docs_dir: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.docs_dir = str(docs_dir or REFERENCE_DOCS_DIR)
        self.db_path = str(db_path or RULE_BOOK_DB_PATH)

        # 初始化经测试验证稳定的 OpenAI 客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(
                TIMEOUT_CONFIG["total_timeout"],
                read=TIMEOUT_CONFIG["read_timeout"],
                write=TIMEOUT_CONFIG.get("write_timeout", 10.0),
                connect=TIMEOUT_CONFIG.get("connect_timeout", 10.0),
            ),
            max_retries=TIMEOUT_CONFIG.get("max_retries", 2),
        )

        self.kb = None
        self.rule_book = None
        self.tool_mapping: Dict[str, Callable[[str], str]] = {}
        self._init_tools()

    def _init_tools(self):
        """初始化知识库与企业法典检索工具"""
        # 工具 1 & 工具 2：PDF 知识库（法规 + 合规手册）
        if PDFKnowledgeBase and os.path.exists(self.docs_dir):
            print(f"[*] 正在从配置目录载入权威 PDF 参考库: {self.docs_dir}", flush=True)
            self.kb = PDFKnowledgeBase(self.docs_dir)
            self.tool_mapping["search_civil_code"] = lambda q: self.kb.search_keyword(q, filter_tag="法")
            self.tool_mapping["get_company_policy"] = lambda q: self.kb.search_keyword(q, filter_tag="合规")
        else:
            print(f"[!] PDF 参考库目录不存在或解析组件缺失，启用法规占位桩。", flush=True)
            self.tool_mapping["search_civil_code"] = lambda q: f"[模拟] 查阅法规关于: {q}"
            self.tool_mapping["get_company_policy"] = lambda q: f"[模拟] 查阅政策关于: {q}"

        # 工具 3：企业自编法典与审查偏好记忆库 (SQLite)
        if EnterpriseRuleBook:
            print(f"[*] 正在连接企业自编法典数据库: {self.db_path}", flush=True)
            self.rule_book = EnterpriseRuleBook(db_path=self.db_path)
            self.tool_mapping["get_past_review_rules"] = lambda q: self.rule_book.search_rules(q)
        else:
            print(f"[!] 未检测到 EnterpriseRuleBook 服务，启用自编法典占位桩。", flush=True)
            self.tool_mapping["get_past_review_rules"] = (
                lambda q: f"《企业自编法典》中暂无针对【{q}】的特殊审查规则。"
            )

    def register_tool(self, name: str, func: Callable[[str], str]):
        """支持向 Agent 动态注入扩展工具"""
        self.tool_mapping[name] = func

    def run_contract_agent(
        self,
        contract_text: str,
        max_turns: Optional[int] = None,
    ) -> AgentExecutionResult:
        """
        执行完整的 ReAct 推理闭环
        :param contract_text: 待审查合同原文
        :param max_turns: 最大步数，默认读取配置中心 AGENT_CONFIG["max_turns"]
        :return: 强类型的 AgentExecutionResult 结果容器
        """
        limit_turns = max_turns or AGENT_CONFIG.get("max_turns", 6)
        user_prompt = USER_CONTRACT_INPUT_TEMPLATE.format(contract_text=contract_text)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print(f"\n>>> 启动合同智能审查流程 (最大轮次限制: {limit_turns})...\n", flush=True)

        for turn in range(limit_turns):
            print(f"\n================ 第 {turn + 1} 轮推理决策 ================", flush=True)

            # 最后一轮施加强收敛约束，禁止继续调用工具
            if turn == limit_turns - 1:
                messages.append({
                    "role": "user",
                    "content": FORCE_FINAL_CONVERGENCE_PROMPT,
                })

            reply_chunks = []
            try:
                stream_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=AGENT_CONFIG.get("temperature", 0.0),
                    top_p=AGENT_CONFIG.get("top_p", 1.0),
                    seed=AGENT_CONFIG.get("seed", 42),
                    max_tokens=AGENT_CONFIG.get("max_tokens", 1024),
                    stop=AGENT_CONFIG.get("stop", ["Observation:"]),
                    stream=True,
                )

                for chunk in stream_response:
                    if not chunk.choices:
                        continue

                    delta_content = getattr(chunk.choices[0].delta, "content", None)
                    if delta_content:
                        sys.stdout.write(delta_content)
                        sys.stdout.flush()
                        reply_chunks.append(delta_content)

                    finish_reason = getattr(chunk.choices[0], "finish_reason", None)
                    if finish_reason == "length":
                        print("\n[警告: 单轮 Token 达到 max_tokens 上限，可能存在截断]", flush=True)

            except httpx.ReadTimeout:
                print("\n[错误: 推理服务响应超时]", flush=True)
                return AgentExecutionResult(
                    status="timeout",
                    turns=turn + 1,
                    report=None,
                    raw_report="",
                    error_message="推理服务端响应超时 (ReadTimeout)",
                )
            except Exception as e:
                print(f"\n[错误: 推理流式交互发生异常: {e}]", flush=True)
                return AgentExecutionResult(
                    status="error",
                    turns=turn + 1,
                    report=None,
                    raw_report=str(e),
                    error_message=str(e),
                )

            reply = "".join(reply_chunks).strip()
            print()

            # 1. 优先检测是否达成 Final 报告，同时兼容模型漏写 Final: 的 Markdown 输出
            is_final, raw_report, parsed_report = split_final_output(reply)
            if is_final:
                print("\n>>> 审查完成，正在解析结构化报告！", flush=True)

                return AgentExecutionResult(
                    status="success" if parsed_report else "partial_success",
                    turns=turn + 1,
                    report=parsed_report,
                    raw_report=raw_report,
                )

            # 2. 解析并执行工具调用
            tool_name, tool_arg = extract_action(reply)
            if tool_name and tool_name in self.tool_mapping:
                print(f"\n[系统执行工具] -> {tool_name}('{tool_arg}')", flush=True)
                try:
                    observation = self.tool_mapping[tool_name](tool_arg)
                except Exception as ex:
                    observation = f"工具执行异常: {str(ex)}"

                print(f"[工具返回结果] ->\n{observation}\n", flush=True)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": format_observation(observation)})
            else:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": TOOL_CALL_RETRY_PROMPT,
                })

        return AgentExecutionResult(
            status="not_converged",
            turns=limit_turns,
            report=None,
            raw_report="审查超时：未在最大轮次内收敛出最终报告。",
        )


def main():
    """本地独立执行测试入口"""
    sample_contract = """
    第一条 乙方逾期交货的，应按合同总金额的 5% 按日向甲方支付违约金。
    第二条 甲方收到货物后，视资金充裕情况安排支付剩余货款。
    第三条 因本合同履行发生的争议，双方一致同意提交甲方住所地当地由甲方指定的仲裁员在其个人办公室内进行独任仲裁，裁决为终局。
    """

    agent = ContractReviewAgent()
    result = agent.run_contract_agent(sample_contract.strip())

    print("\n" + "=" * 60)
    print(f"审查结果状态: {result.status} (实际耗费轮次: {result.turns})")
    print("=" * 60)

    if result.report:
        print(f"结构化审查项总计: {len(result.report.reviews)} 条")
        print(f"风险分布: 高风险({result.report.high_risk_count}) | 中风险({result.report.medium_risk_count}) | 低风险({result.report.low_risk_count})")
        for idx, item in enumerate(result.report.reviews, 1):
            print(f"\n[{idx}] 审查主题: {item.clause_topic} | 风险等级: {item.risk_level.value}")
            print(f"    检索依据: {item.legal_basis}")
            print(f"    风险剖析: {item.issue}")
            print(f"    修改建议: {item.suggested_revision}")
    else:
        print("\n>>> 原始审查输出:\n", result.raw_report)


if __name__ == "__main__":
    main()