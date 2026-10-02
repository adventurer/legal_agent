"""Opt-in live test for model use of directly injected enterprise KB evidence."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace

from openai import OpenAI

from configs.config import DEFAULT_MODEL_NAME, VLLM_API_KEY, VLLM_BASE_URL
from core.prompts import RISK_GRADING_GUIDANCE
from gateway.llm_interaction.react_loop import ReactLoop


@unittest.skipUnless(
    os.getenv("RUN_LIVE_MODEL_TESTS") == "1",
    "Set RUN_LIVE_MODEL_TESTS=1 to call the local vLLM model.",
)
class LiveKnowledgeBaseAdoptionTests(unittest.TestCase):
    def test_model_applies_directly_injected_payment_rule(self):
        contract_text = "甲方应于验收合格后60个工作日内向乙方支付尾款。"
        evidence = {
            "id": "RULE2",
            "source_type": "enterprise_rule",
            "source_location": "data/rule_book.db · 规则 RULE2",
            "title": "付款条件与结算周期",
            "enterprise_risk_level": "High",
            "text": (
                "审查标准：尾款支付周期自验收合格起不得超过30个工作日。\n"
                "禁止情形：无确定期限的付款条款；超过90个工作日的过长付款周期。\n"
                "建议条款：验收合格并收到发票后15个工作日内支付尾款。"
            ),
        }
        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=90.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=DEFAULT_MODEL_NAME,
            temperature=0.0,
            seed=42,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是合同审查助手。只根据用户提供的合同原文和企业知识库证据作答。"
                        "企业规则属于内部要求，不得伪装成法律规定。\n"
                        f"{RISK_GRADING_GUIDANCE}\n"
                        "JSON 中 applicability 必须填写 applicable 或 not_applicable。"
                        "只要规则所管事项出现在合同中就填 applicable；合同违反规则也仍然适用。"
                        "只有合同完全不涉及规则所管事项时才填 not_applicable。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "审查以下合同条款，并仅返回 JSON，字段为 applicability (applicable/not_applicable)、"
                        "enterprise_risk_level、evidence_reference、reason、recommendation。"
                        "evidence_reference 必须严格使用 [[RULE:RULE2]] 格式。\n"
                        f"合同原文：{contract_text}\n"
                        "企业知识库检索结果：\n"
                        f"{json.dumps(evidence, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        output = response.choices[0].message.content or ""

        payload_start = output.find("{")
        payload_end = output.rfind("}")
        self.assertGreaterEqual(payload_start, 0, output)
        result = json.loads(output[payload_start:payload_end + 1])

        self.assertEqual(result["applicability"], "applicable")
        self.assertEqual(result["enterprise_risk_level"], "High")
        self.assertEqual(result["evidence_reference"], "[[RULE:RULE2]]")
        self.assertRegex(result["reason"], r"30|三十")
        self.assertTrue(result["recommendation"])

    def test_react_loop_uses_pinned_enterprise_rule_in_final_report(self):
        contract_text = (
            "协商不成时，争议提交甲方指定的独任仲裁员在其个人办公场所仲裁，"
            "任何一方不得向人民法院起诉或提出管辖权异议。"
        )
        rule = {
            "id": "RULE3",
            "source_type": "enterprise_rule",
            "source_location": "data/rule_book.db · 规则 RULE3",
            "title": "争议管辖与仲裁机构",
            "enterprise_risk_level": "High",
            "text": (
                "审查标准：严禁接受单方指定仲裁员、单方确定管辖地或非正规民间仲裁。"
                "原则上约定我方所在地人民法院管辖，或正规仲裁委员会。\n"
                "禁止情形：由对方单方指定独任仲裁员在其办公室内仲裁。"
            ),
        }

        def law_search(query):
            return json.dumps({"evidence": [{
                "id": "EV_LAW_TEST",
                "source_type": "law",
                "article_no": "第八十一条",
                "source_location": "测试法规库",
                "text": "当事人可以书面约定仲裁地。",
            }]}, ensure_ascii=False)

        def company_policy(query):
            return json.dumps({"evidence": [rule]}, ensure_ascii=False)

        def past_rules(query):
            return json.dumps({"evidence": [rule]}, ensure_ascii=False)

        def general_materials(query):
            return json.dumps({"evidence": []}, ensure_ascii=False)

        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=90.0,
            max_retries=0,
        )
        agent = SimpleNamespace(
            model_name=DEFAULT_MODEL_NAME,
            client=client,
            tool_mapping={
                "search_civil_code": law_search,
                "get_company_policy": company_policy,
                "get_past_review_rules": past_rules,
                "search_general_materials": general_materials,
            },
            tool_registry=SimpleNamespace(unavailable_tools=set()),
        )

        async def collect():
            return [event async for event in ReactLoop(agent).stream(
                contract_text,
                10,
                "live-kb-adoption",
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )

        self.assertEqual(final["status"], "success")
        self.assertIn("[[RULE:RULE3]]", final["raw_report"])
        self.assertNotIn("未检索到相关企业规则", final["raw_report"])
        self.assertRegex(final["raw_report"], r"人民法院|仲裁委员会")


if __name__ == "__main__":
    unittest.main()
