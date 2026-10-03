"""Contract tests for the native-tool-call LLM interaction layer."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from configs import config as app_config
from core.schemas import ReviewRequest
from gateway.llm_interaction.context_manager import ContextWindowManager
from gateway.llm_interaction.react_loop import _max_output_tokens
from gateway.llm_interaction.contracts import ModelToolCall
from gateway.llm_interaction.events import encode_event
from gateway.llm_interaction.final_report import (
    FinalReportValidationError,
    validate_final_report,
)
from gateway.llm_interaction.guardrails import detect_contract_flags
from gateway.llm_interaction.react_loop import ReactLoop
from gateway.llm_interaction.prompt_builder import build_review_messages
from gateway.llm_interaction.report_finalizer import ReportFinalizer
from gateway.llm_interaction.tool_catalog import build_tool_schemas
from gateway.llm_interaction.tool_executor import ToolExecutor
from gateway.llm_interaction.tool_call_recorder import ToolCallRecorder
from core.prompts import (
    AGENT_SYSTEM_PROMPT,
    EMPTY_SEARCH_RETRY_PROMPT,
    FINAL_CONSISTENCY_CHECK_PROMPT,
    REPORT_OUTPUT_GUIDANCE,
    REVIEW_ANALYSIS_GUIDANCE,
    RISK_LEVEL_REPORT_LEGEND,
)
from core.schemas import ContractReviewReport
from services.pdf_kb_search import search_pages
from services.report_parser import parse_structured_report


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    choices = []
    if content is not None or tool_calls is not None or finish_reason is not None:
        choices.append(SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        ))
    return SimpleNamespace(choices=choices, usage=usage)


def _tool_call_delta(call_id=None, name=None, arguments=None, index=0):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _valid_report_payload(**review_overrides):
    review = {
        "clause_topic": "第五条 验收",
        "risk_level": "Medium",
        "risk_type": "商业",
        "enterprise_risk_level": "未检索到企业内部风险等级",
        "legal_effect": "未检索到直接依据",
        "commercial_impact": "验收流程不明确，可能导致付款争议。",
        "remedy_cost": "中",
        "affected_party": "双方",
        "confidence": "中",
        "legal_basis": "未检索到直接依据",
        "enterprise_basis": "未检索到相关企业规则或知识库依据",
        "issue": "验收期限未明确。",
        "suggested_revision": "明确验收期限。",
    }
    review.update(review_overrides)
    return {"reviews": [review]}


def _canonical_report_json(report_payload=None):
    payload = report_payload or _valid_report_payload()
    return ContractReviewReport.model_validate(payload).model_dump_json(ensure_ascii=False)


def _final_report_arguments(report_payload=None, guardrails=None):
    return json.dumps({
        "report": report_payload or _valid_report_payload(),
        "acknowledged_guardrails": guardrails or [],
    }, ensure_ascii=False)


def _agent(responses, tool_mapping=None):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return iter(next(responses))

    agent = SimpleNamespace(
        model_name="mock-model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        tool_mapping=tool_mapping or {},
        tool_registry=SimpleNamespace(unavailable_tools=set()),
    )
    return agent, requests


class ToolContractTests(unittest.TestCase):
    def test_review_request_accepts_ui_max_turns_and_rejects_above_it(self):
        contract = "这是用于验证最大探索步数的合同正文样例，长度超过十个字符。"

        request = ReviewRequest(contract_text=contract, max_turns=30)

        self.assertEqual(request.max_turns, 30)
        with self.assertRaises(ValueError):
            ReviewRequest(contract_text=contract, max_turns=31)

    def test_catalog_is_bounded_and_excludes_unavailable_tools(self):
        mapping = {
            name: (lambda query: query)
            for name in (
                "search_civil_code",
                "get_company_policy",
                "get_past_review_rules",
                "search_general_materials",
                "unregistered_write_tool",
            )
        }
        schemas = build_tool_schemas(mapping, {"get_company_policy"})
        names = [item["function"]["name"] for item in schemas]
        self.assertEqual(len(schemas), 4)
        self.assertNotIn("get_company_policy", names)
        self.assertNotIn("unregistered_write_tool", names)
        self.assertEqual(names[-1], "submit_final_report")
        rule_description = next(
            item["function"]["description"]
            for item in schemas
            if item["function"]["name"] == "get_past_review_rules"
        )
        self.assertIn("风险等级", rule_description)
        self.assertIn("审查标准", rule_description)
        self.assertIn("禁止情形", rule_description)
        self.assertIn("不得将企业规则作为法律依据", rule_description)
        self.assertFalse(schemas[0]["function"]["parameters"]["additionalProperties"])
        self.assertIn("不得填写工具名", schemas[-1]["function"]["description"])
        self.assertIn("可省略", schemas[-1]["function"]["description"])
        report_parameters = schemas[-1]["function"]["parameters"]
        self.assertNotIn("acknowledged_guardrails", report_parameters["required"])

        guarded_schemas = build_tool_schemas(
            mapping,
            {"get_company_policy"},
            ["unlimited_liability"],
        )
        guarded_required = guarded_schemas[-1]["function"]["parameters"]["required"]
        self.assertIn("acknowledged_guardrails", guarded_required)

    def test_executor_validates_arguments_before_read_only_call(self):
        calls = []
        executor = ToolExecutor({
            "search_civil_code": lambda query: calls.append(query) or "evidence",
            "write_file": lambda query: "must not be exposed",
        })
        invalid = executor.execute(ModelToolCall(
            call_id="call_bad",
            name="search_civil_code",
            arguments='{"query":"civil","shell":"true"}',
        ))
        self.assertFalse(invalid.success)
        self.assertEqual(calls, [])

        valid = executor.execute(ModelToolCall(
            call_id="call_ok",
            name="search_civil_code",
            arguments='{"query":" civil code "}',
        ))
        self.assertTrue(valid.success)
        self.assertEqual(valid.query, "civil code")
        self.assertEqual(calls, ["civil code"])

    def test_executor_blocks_normalized_duplicate_queries(self):
        calls = []
        executor = ToolExecutor({
            "search_civil_code": lambda query: calls.append(query) or "evidence",
        })
        first = executor.execute(ModelToolCall(
            call_id="first",
            name="search_civil_code",
            arguments='{"query":"买卖合同 检验期限"}',
        ))
        repeated = executor.execute(ModelToolCall(
            call_id="repeat",
            name="search_civil_code",
            arguments='{"query":" 买卖合同，检验期限 "}',
        ))
        self.assertTrue(first.success)
        self.assertFalse(repeated.success)
        self.assertTrue(repeated.duplicate)
        self.assertEqual(len(calls), 1)

    def test_search_ignores_generic_terms_and_unrelated_pages(self):
        pages = [{
            "id": "EV1",
            "doc_name": "arbitration.txt",
            "tag": "法",
            "page_num": 1,
            "text": "中华人民共和国仲裁法。中国法规由相关部门发布，法律依据应当检索。",
        }]
        result = json.loads(search_pages(
            pages,
            "验收标准 付款周期 合同法 中国 法规",
            filter_tag="法",
        ))
        self.assertEqual(result["evidence"], [])

    def test_search_returns_specific_matching_law(self):
        pages = [{
            "id": "EV621",
            "doc_name": "civil_code.txt",
            "tag": "法",
            "page_num": 1,
            "article_no": "第六百二十一条",
            "text": "买卖合同中，买受人应当在检验期限内通知出卖人数量或者质量不符合约定。",
        }]
        result = json.loads(search_pages(
            pages,
            "买卖合同 检验期限",
            filter_tag="法",
        ))
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["article_no"], "第六百二十一条")

    def test_composite_arbitration_query_returns_pages_matching_two_facets(self):
        pages = [
            {
                "id": "EV-ARBITRATION",
                "doc_name": "仲裁法.txt",
                "tag": "法",
                "page_num": 1,
                "text": "仲裁案件由仲裁委员会受理，仲裁委员会依照法律规定行使管辖权。",
            },
            {
                "id": "EV-ONE-TERM",
                "doc_name": "other-law.txt",
                "tag": "法",
                "page_num": 2,
                "text": "合同中应当明确争议解决方式，仲裁是可选方式之一。",
            },
        ]

        result = json.loads(search_pages(
            pages,
            "仲裁 管辖 地点 费用 败诉方",
            filter_tag="法",
        ))

        self.assertEqual(
            [item["id"] for item in result["evidence"]],
            ["EV-ARBITRATION"],
        )

    def test_final_report_requires_guardrail_acknowledgments(self):
        call = ModelToolCall(
            call_id="final",
            name="submit_final_report",
            arguments=_final_report_arguments(),
        )
        with self.assertRaises(FinalReportValidationError):
            validate_final_report(call, ["unlimited_liability"])

    def test_missing_guardrail_field_defaults_only_when_none_are_required(self):
        report_payload = _valid_report_payload()
        nested_report = {
            **report_payload,
            "acknowledged_guardrails": [],
        }
        call = ModelToolCall(
            call_id="final",
            name="submit_final_report",
            arguments=json.dumps({"report": report_payload}, ensure_ascii=False),
        )
        nested_call = ModelToolCall(
            call_id="nested-final",
            name="submit_final_report",
            arguments=json.dumps({"report": nested_report}, ensure_ascii=False),
        )

        submission = validate_final_report(call, [])
        nested_submission = validate_final_report(nested_call, [])

        self.assertEqual(submission.acknowledged_guardrails, [])
        self.assertEqual(nested_submission.acknowledged_guardrails, [])
        with self.assertRaisesRegex(
            FinalReportValidationError,
            "遗漏必须核实的风险标记: unlimited_liability",
        ):
            validate_final_report(call, ["unlimited_liability"])
        with self.assertRaisesRegex(
            FinalReportValidationError,
            "遗漏必须核实的风险标记: unlimited_liability",
        ):
            validate_final_report(nested_call, ["unlimited_liability"])

class ContextAndGuardrailTests(unittest.TestCase):
    def test_qwen_1_5b_budget_matches_its_context_and_output_ceiling(self):
        model_context = app_config.MODEL_PRESETS["qwen2.5-1.5b-awq"]["max_model_len"]
        requested_output = app_config.AGENT_CONFIG["max_tokens"]
        actual_output = _max_output_tokens(model_context, requested_output)

        context = ContextWindowManager(
            "unknown-model",
            max_context_tokens=model_context,
            output_tokens=actual_output,
        )
        self.assertEqual(
            context._max_input_tokens,
            int(model_context * 0.98) - actual_output,
        )
        self.assertEqual(_max_output_tokens(32768, 8192), 8192)
        self.assertEqual(_max_output_tokens(8192, 8192), 2048)
        self.assertEqual(_max_output_tokens(4096, 8192), 1024)

    def test_context_budget_leaves_two_percent_for_runtime_overhead(self):
        context = ContextWindowManager(
            "unknown-model",
            max_context_tokens=8192,
            output_tokens=2048,
        )

        self.assertEqual(context._max_input_tokens, 5980)

    def test_context_window_drops_complete_old_tool_turns(self):
        context = ContextWindowManager(
            "unknown-model",
            max_context_tokens=8192,
            output_tokens=512,
            max_history_turns=1,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "contract"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "old"}]},
            {"role": "tool", "tool_call_id": "old", "content": "old result"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "new"}]},
            {"role": "tool", "tool_call_id": "new", "content": "new result"},
        ]
        fitted = context.fit_messages(messages, [])
        self.assertEqual([item.get("tool_call_id") for item in fitted if item["role"] == "tool"], ["new"])
        self.assertEqual(fitted[:2], messages[:2])

    def test_context_window_preserves_enterprise_evidence_after_duplicate_searches(self):
        context = ContextWindowManager(
            "unknown-model",
            max_context_tokens=8192,
            output_tokens=512,
            max_history_turns=4,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "contract"},
        ]

        def append_tool_turn(call_id, tool_name, observation):
            messages.extend([
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "function": {"name": tool_name},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(observation),
                },
            ])

        append_tool_turn("law", "search_civil_code", {
            "evidence": [{"id": "LAW_OLD", "source_type": "law"}],
        })
        append_tool_turn("company", "get_company_policy", {
            "evidence": [
                {"id": "DOC1", "source_type": "enterprise_document"},
                {"id": "RULE2", "source_type": "enterprise_rule"},
            ],
        })
        append_tool_turn("rules", "get_past_review_rules", {
            "evidence": [{"id": "RULE2", "source_type": "enterprise_rule"}],
        })
        append_tool_turn("general", "search_general_materials", {"evidence": []})
        for call_id in ("duplicate1", "duplicate2", "duplicate3"):
            append_tool_turn(call_id, "search_civil_code", {
                "ok": False,
                "error": "重复检索",
            })

        fitted = context.fit_messages(messages, [])
        tool_content = "\n".join(
            message["content"]
            for message in fitted
            if message["role"] == "tool"
        )

        self.assertIn("RULE2", tool_content)
        self.assertIn("DOC1", tool_content)
        self.assertNotIn("LAW_OLD", tool_content)
        self.assertIn("duplicate3", [
            message.get("tool_call_id")
            for message in fitted
            if message["role"] == "tool"
        ])

    def test_deterministic_financial_flags(self):
        findings = detect_contract_flags(
            "甲方每日按总价0.1%支付逾期金。乙方每日按总价1.2%支付违约金。"
            "乙方赔偿责任不设责任上限。"
        )
        self.assertEqual(
            {finding.code for finding in findings},
            {"daily_rate_asymmetry", "unlimited_liability"},
        )


class PromptAlignmentTests(unittest.TestCase):
    def test_prompts_forbid_fallback_to_pretrained_knowledge(self):
        messages = build_review_messages("合同正文", "neutral", [])
        prompts = (
            messages[0]["content"],
            AGENT_SYSTEM_PROMPT,
            FINAL_CONSISTENCY_CHECK_PROMPT,
            EMPTY_SEARCH_RETRY_PROMPT,
        )
        for prompt in prompts:
            self.assertIn("知识库", prompt)
            self.assertRegex(prompt, r"不得.{0,12}(?:模型记忆|预训练记忆|专业常识)")
        self.assertNotIn("直接使用模型已有法律知识", EMPTY_SEARCH_RETRY_PROMPT)

    def test_native_prompt_includes_shared_review_requirements(self):
        messages = build_review_messages("第一条 合同期限", "buyer", [])
        system_prompt = messages[0]["content"]

        self.assertIn(REVIEW_ANALYSIS_GUIDANCE, system_prompt)
        self.assertIn(REPORT_OUTPUT_GUIDANCE, system_prompt)
        self.assertIn("逾期 30 日", system_prompt)
        self.assertIn("违反规则应报告为风险，不能因此判为不适用", system_prompt)
        self.assertIn("不得写成“未检索到相关企业规则”", system_prompt)
        self.assertIn("[[EVIDENCE:EV编号]]", system_prompt)
        self.assertIn("submit_final_report", system_prompt)
        self.assertIn("不得填写工具名", system_prompt)
        self.assertNotIn("工具轮只输出 Thought 和一行 Action", system_prompt)
        self.assertNotIn("最终轮以 Thought 和 Final: 开始", system_prompt)


class ReActLoopTests(unittest.TestCase):
    def test_enterprise_rule_assessments_require_complete_coverage(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

    ### 第二条 付款条件
    - **风险类型**: 商业
    - **风险等级**: 中风险
    - **企业内部风险等级**: 未检索到企业内部风险等级
    - **法律效力**: 未检索到直接依据
    - **商业后果**: 付款期限不明确可能导致结算延迟。
    - **救济成本**: 中
    - **受影响方**: 乙方
    - **结论置信度**: 中
    - **法律/合规依据**: 未检索到直接依据
    - **企业知识库依据**: 未检索到相关企业规则或知识库依据
    - **风险剖析**: 付款期限需要明确。
    - **修改建议**: 明确付款期限。
    """
        evidence = {
            "RULE_MEDIUM": {
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "Medium",
            },
            "RULE_LOW": {
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "Low",
            },
        }
        assessments = [
            {
                "evidence_id": "RULE_MEDIUM",
                "applicability": "applicable",
                "applicable_clauses": [{
                    "clause_topic": "第二条 付款条件",
                    "contract_basis": "验收后 60 个工作日付款",
                    "reason": "超过规则允许的付款周期。",
                    "recommendation_for_report": "将付款周期改为 30 个工作日内。",
                }],
            },
            {
                "evidence_id": "RULE_LOW",
                "applicability": "not_applicable",
                "applicable_clauses": [],
                "contract_basis": "验收后 60 个工作日付款",
                "not_applicable_reason": "合同没有涉及该规则约束的事项。",
            },
        ]

        proposals, issues = ReportFinalizer._validated_rule_assessments(
            report, "验收后 60 个工作日付款", assessments, evidence
        )
        self.assertEqual(issues, [])
        self.assertEqual(
            proposals["第二条 付款条件"]["evidence_ids"],
            ["RULE_MEDIUM"],
        )
        self.assertEqual(
            proposals["第二条 付款条件"]["recommendation_adoptions"],
            ["将付款周期改为 30 个工作日内。"],
        )
        self.assertEqual(
            proposals["第二条 付款条件"]["rule_impacts"][0]["enterprise_risk_level"],
            "Medium",
        )

        _, missing_issues = ReportFinalizer._validated_rule_assessments(
            report, "验收后 60 个工作日付款", assessments[:1], evidence
        )
        self.assertIn("规则 RULE_LOW 未核对", missing_issues)

        unsupported = [dict(assessments[1], contract_basis="验收后 15 个工作日付款")]
        _, unsupported_issues = ReportFinalizer._validated_rule_assessments(
            report, "验收后 60 个工作日付款", unsupported, {"RULE_LOW": evidence["RULE_LOW"]}
        )
        self.assertIn("不适用判断没有可在合同原文中核实的引文", unsupported_issues[0])

    def test_applicable_enterprise_risk_grade_sets_non_downgrading_floor(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

### 第二条 付款条件
- **风险类型**: 商业
- **风险等级**: 低风险
- **企业内部风险等级**: 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 付款期限不明确。
- **救济成本**: 低
- **受影响方**: 乙方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 付款期限需要明确。
- **修改建议**: 明确付款期限。
"""
        expected_floors = {
            "High": "高风险",
            "Medium": "中风险",
            "Low": "低风险",
            "Notice": "提示",
        }
        for evidence_id, (enterprise_grade, expected_risk) in enumerate(
            expected_floors.items(), start=1
        ):
            rule_id = f"RULE{evidence_id}"
            evidence = {
                rule_id: {
                    "source_type": "enterprise_rule",
                    "source_location": f"rules.db · {rule_id}",
                    "enterprise_risk_level": enterprise_grade,
                }
            }
            applied = ReportFinalizer._validated_supplements(
                report,
                {"第二条 付款条件": {
                    "evidence_ids": [rule_id],
                    "rule_impacts": [{
                        "evidence_id": rule_id,
                        "enterprise_risk_level": enterprise_grade,
                        "contract_basis": "验收后 60 个工作日付款",
                        "reason": "超过企业规则规定的付款周期。",
                    }],
                    "recommendation_adoptions": ["改为 30 个工作日内付款。"],
                }},
                evidence,
            )
            if expected_risk in {"高风险", "中风险"}:
                self.assertEqual(
                    applied["第二条 付款条件"]["risk_level"],
                    expected_risk,
                )
            else:
                self.assertNotIn("risk_level", applied["第二条 付款条件"])
            self.assertIn(
                "企业规则核对：",
                applied["第二条 付款条件"]["issue"],
            )

        high_report = report.replace("- **风险等级**: 低风险", "- **风险等级**: 高风险")
        low_rule = {
            "RULE_LOW": {
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "Low",
            }
        }
        preserved = ReportFinalizer._validated_supplements(
            high_report,
            {"第二条 付款条件": {
                "evidence_ids": ["RULE_LOW"],
                "rule_impacts": [{
                    "evidence_id": "RULE_LOW",
                    "enterprise_risk_level": "Low",
                    "contract_basis": "合同事实",
                    "reason": "规则适用。",
                }],
            }},
            low_rule,
        )
        self.assertNotIn("risk_level", preserved["第二条 付款条件"])

    def test_applicable_high_enterprise_rule_forces_high_risk(self):
        report = """### 第二条 付款条件
- **风险等级**: 低风险
- **法律/合规依据**: 未检索到直接依据
- **风险剖析**: 付款条件待核验。
- **修改建议**: 明确付款期限。
"""
        evidence = {
            "RULE2": {
                "id": "RULE2",
                "source_type": "enterprise_rule",
                "source_location": "data/rule_book.db · 规则 RULE2",
                "enterprise_risk_level": "High",
            },
            "RULE9": {
                "id": "RULE9",
                "source_type": "enterprise_rule",
                "source_location": "data/rule_book.db · 规则 RULE9",
                "enterprise_risk_level": "High",
            },
        }

        applicable = ReportFinalizer._validated_supplements(
            report,
            {"第二条 付款条件": {
                "evidence_ids": ["RULE2"],
                "raise_to_high_risk": False,
            }},
            evidence,
        )
        unrelated = ReportFinalizer._validated_supplements(
            report,
            {"第二条 付款条件": {
                "evidence_ids": [],
                "raise_to_high_risk": True,
            }},
            evidence,
        )

        self.assertEqual(applicable["第二条 付款条件"]["risk_level"], "高风险")
        self.assertEqual(applicable["第二条 付款条件"]["enterprise_risk_level"], "High")
        self.assertNotIn("第二条 付款条件", unrelated)

    def test_finalizer_passes_through_report_without_validation(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

### 第五条 服务保障
- **风险类型**: 商业
- **风险等级**: 中风险
- **企业内部风险等级**: 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 验收流程不明确，可能导致付款争议。
- **救济成本**: 中
- **受影响方**: 乙方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 验收期限未明确，付款条件可能长期无法触发。
- **修改建议**: 明确验收期限和逾期未反馈的处理方式。
"""
        agent, requests = _agent(iter([]))
        recorder = ToolCallRecorder("valid-report-format-check")

        async def collect():
            return [event async for event in ReportFinalizer(agent).finalize(
                report=report,
                contract_text="第五条约定验收后付款，但未约定验收期限。",
                task_id="第五条 服务保障",
                turn=1,
                acknowledged_guardrails=[],
                finish_reason="tool_calls",
                tool_call_records=recorder,
            )]

        with patch.dict(app_config.AGENT_CONFIG, {"report_format_repair_enabled": False}):
            events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertEqual(requests, [])
        self.assertFalse(any(
            event["event"] == "pipeline_stage"
            and json.loads(event["data"])["stage"] == "format_validation"
            for event in events
        ))
        self.assertEqual(final["status"], "success")
        self.assertEqual(final["raw_report"], report)
        self.assertEqual(requests, [])

    def test_finalizer_restores_authoritative_enterprise_risk_level(self):
        report = _valid_report_payload(
            enterprise_risk_level="Low",
            enterprise_basis="规则库 [[RULE:RULE7]]",
        )
        agent, _ = _agent(iter([]))
        recorder = ToolCallRecorder("authoritative-enterprise-grade")
        recorder.record_call(1, "call_rule", "get_past_review_rules", "{}")
        recorder.record_result(
            "call_rule",
            json.dumps({"evidence": [{
                "id": "RULE7",
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "High",
            }]}, ensure_ascii=False),
            True,
            0,
        )

        async def collect():
            return [event async for event in ReportFinalizer(agent).finalize(
                report=json.dumps(report, ensure_ascii=False),
                contract_text="合同条款内容",
                task_id="authoritative-enterprise-grade",
                turn=1,
                acknowledged_guardrails=[],
                finish_reason="tool_calls",
                tool_call_records=recorder,
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        final_report = json.loads(final["raw_report"])
        self.assertEqual(
            final_report["reviews"][0]["enterprise_risk_level"],
            "High",
        )
        self.assertEqual(final_report["reviews"][0]["risk_level"], "Medium")

    def test_finalizer_passes_through_report_without_rule_reconciliation(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

    ### 第二条 付款条件
     - **风险类型**: 商业
     - **风险等级**: 中风险
     - **企业内部风险等级**: 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 付款期限不明确可能导致结算延迟。
- **救济成本**: 中
- **受影响方**: 乙方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 付款期限需要明确。
- **修改建议**: 明确付款期限。
"""
        agent, requests = _agent(iter([]))
        recorder = ToolCallRecorder("missing-rule-assessment")
        recorder.record_call(1, "call_rule", "get_past_review_rules", "{\"query\":\"付款期限\"}")
        recorder.record_result(
            "call_rule",
            json.dumps({"evidence": [{
                "id": "RULE2",
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "Medium",
                "source_location": "data/rule_book.db · 规则 RULE2",
                "text": "付款周期不得超过 30 个工作日。",
            }]}),
            True,
            0,
        )

        async def collect():
            return [event async for event in ReportFinalizer(agent).finalize(
                report=report,
                contract_text="验收合格后 60 个工作日付款。",
                task_id="第二条 付款条件",
                turn=2,
                acknowledged_guardrails=[],
                finish_reason="tool_calls",
                tool_call_records=recorder,
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertFalse(any(event["event"] == "rule_assessment" for event in events))
        self.assertEqual(final["raw_report"], report)
        self.assertFalse(any(
            event["event"] == "pipeline_stage"
            and json.loads(event["data"])["stage"] == "report_reconciliation"
            for event in events
        ))
        self.assertEqual(final["status"], "success")
        self.assertTrue(final["is_complete"])
        self.assertEqual(recorder.status, "success")
        self.assertEqual(requests, [])

    def test_finalizer_directly_outputs_report_with_enterprise_rule(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

### 第二条 付款条件
- **风险类型**: 商业
- **风险等级**: 低风险
- **企业内部风险等级**: 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 付款条件存在不确定性。
- **救济成本**: 中
- **受影响方**: 乙方
- **结论置信度**: 中
- **法律/合规依据**: [[RULE:RULE2]]
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 尾款付款时间由甲方资金情况决定。
- **修改建议**: 明确尾款支付期限。
"""
        contract_text = "甲方按季度资金周转充裕情况安排支付尾款，且付款期限不受商业惯例限制。"
        agent, requests = _agent(iter([]))
        recorder = ToolCallRecorder("contradictory-rule-assessment")
        recorder.record_call(1, "call_rule", "get_past_review_rules", "{}");
        recorder.record_result(
            "call_rule",
            json.dumps({"evidence": [{
                "id": "RULE2",
                "source_type": "enterprise_rule",
                "enterprise_risk_level": "High",
                "source_location": "data/rule_book.db · 规则 RULE2",
                "text": "禁止视甲方资金情况安排付款。",
            }]}),
            True,
            0,
        )

        async def collect():
            return [event async for event in ReportFinalizer(agent).finalize(
                report=report,
                contract_text=contract_text,
                task_id="第二条 付款条件",
                turn=2,
                acknowledged_guardrails=[],
                finish_reason="tool_calls",
                tool_call_records=recorder,
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertEqual(final["raw_report"], report)
        self.assertIn("[[RULE:RULE2]]", final["raw_report"])
        self.assertEqual(final["status"], "success")
        self.assertTrue(final["is_complete"])
        self.assertEqual(requests, [])

    def test_finalizer_retries_existing_placeholder_fields(self):
        report = f"""{RISK_LEVEL_REPORT_LEGEND}

### 第五条 服务保障
- **风险类型**: 初稿未提供
- **风险等级**: 中风险
- **企业内部风险等级**: 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 验收流程不明确，可能导致付款争议。
- **救济成本**: 中
- **受影响方**: 乙方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 验收期限未明确，付款条件可能长期无法触发。
- **修改建议**: 明确验收期限和逾期未反馈的处理方式。
"""
        repaired_report = report.replace(
            "- **风险类型**: 初稿未提供",
            "- **风险类型**: 商业",
        )
        agent, requests = _agent(iter([]))
        recorder = ToolCallRecorder("placeholder-repair")

        async def collect():
            return [event async for event in ReportFinalizer(agent).finalize(
                report=report,
                contract_text="第五条约定验收后付款，但未约定验收期限。",
                task_id="第五条 服务保障",
                turn=1,
                acknowledged_guardrails=[],
                finish_reason="tool_calls",
                tool_call_records=recorder,
            )]

        with patch.dict(app_config.AGENT_CONFIG, {"report_format_repair_enabled": True}):
            events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertEqual(final["raw_report"], report)
        self.assertIn("初稿未提供", final["raw_report"])
        self.assertEqual(requests, [])

    def test_native_tool_call_returns_role_tool_then_structured_final(self):
        report_payload = _valid_report_payload(
            suggested_revision="建议明确付款期限。"
        )
        report = _canonical_report_json(report_payload)
        report_arguments = _final_report_arguments(report_payload)
        split = report_arguments.index("付款") + 3
        responses = iter([
            [
                _chunk(tool_calls=[_tool_call_delta(
                    "call_search",
                    "search_civil_code",
                    '{"query":"付款期限"}',
                )], finish_reason="tool_calls"),
                _chunk(usage=SimpleNamespace(prompt_tokens=90, completion_tokens=12)),
            ],
            [
                _chunk(tool_calls=[_tool_call_delta(
                    "call_final",
                    "submit_final_report",
                    report_arguments[:split],
                )], finish_reason=None),
                _chunk(tool_calls=[_tool_call_delta(
                    arguments=report_arguments[split:],
                )], finish_reason="tool_calls"),
                _chunk(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=25)),
            ],
        ])
        agent, requests = _agent(
            responses,
            {"search_civil_code": lambda query: json.dumps({
                "evidence": [{"id": "EV1", "text": query, "snippet": query}],
            }, ensure_ascii=False)},
        )

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 4, "task"
            )]

        events = asyncio.run(collect())
        names = [item["event"] for item in events]
        payloads = [json.loads(item["data"]) for item in events]
        first_model_start = names.index("model_start")
        self.assertEqual(payloads[first_model_start]["turn"], 1)
        model_tool_calls = [
            payload for name, payload in zip(names, payloads)
            if name == "model_tool_call"
        ]
        self.assertEqual(len(model_tool_calls), 2)
        self.assertEqual(model_tool_calls[0]["arguments"], '{"query":"付款期限"}')
        self.assertEqual(model_tool_calls[1]["tool"], "submit_final_report")
        submitted_report = json.loads(model_tool_calls[1]["arguments"])["report"]
        self.assertEqual(
            submitted_report["reviews"][0]["suggested_revision"],
            "建议明确付款期限。",
        )
        self.assertEqual(names[-2:], ["final_report", "done"])
        self.assertEqual(names.count("tool_start"), 1)
        self.assertEqual(names.count("tool_result"), 1)
        self.assertEqual(payloads[-2]["raw_report"], report)
        self.assertEqual(
            requests[0]["tool_choice"],
            {"type": "function", "function": {"name": "search_civil_code"}},
        )
        self.assertFalse(requests[0]["parallel_tool_calls"])
        tool_message = requests[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_search")
        self.assertIn("EV1", tool_message["content"])
        tool_result = next(
            payload for name, payload in zip(names, payloads)
            if name == "tool_result"
        )
        self.assertEqual(tool_result["injected_chars"], len(tool_message["content"]))

    def test_enterprise_evidence_guidance_pins_authoritative_grade(self):
        evidence = {
            "RULE7": {
                "source_type": "enterprise_rule",
                "title": "付款周期规则",
                "enterprise_risk_level": "High",
                "text": "风险等级: High\n付款周期不得超过 30 日。",
            }
        }

        guidance = ReactLoop._enterprise_evidence_guidance(evidence)

        pinned_records = json.loads(guidance.split("\n", 1)[1])
        self.assertEqual(pinned_records[0]["enterprise_risk_level"], "High")
        self.assertNotIn("risk_level", pinned_records[0])
        self.assertNotIn("风险等级: High", pinned_records[0]["text"])

    def test_finalization_preserves_model_report_with_enterprise_evidence(self):
        report_payload = _valid_report_payload(
            clause_topic="8.2 仲裁管辖约定",
            risk_level="High",
            legal_basis="[[RULE:RULE3]]",
        )
        report = _canonical_report_json(report_payload)
        rule_evidence = {
            "evidence": [{
                "id": "RULE3",
                "doc_name": "企业自编法典",
                "title": "争议管辖与仲裁机构",
                "source_type": "enterprise_rule",
                "source_location": "data/rule_book.db · 规则 RULE3",
                "enterprise_risk_level": "High",
                "text": "禁止对方单方指定仲裁员。",
            }],
            "message": "",
        }
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                "company_policy",
                "get_company_policy",
                '{"query":"仲裁员选任"}',
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "final",
                "submit_final_report",
                json.dumps({
                    "report": report_payload,
                    "acknowledged_guardrails": [],
                }, ensure_ascii=False),
            )], finish_reason="tool_calls")],
        ])
        agent, requests = _agent(
            responses,
            {"get_company_policy": lambda query: json.dumps(
                rule_evidence, ensure_ascii=False
            )},
        )

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "8.2 甲方单方指定独任仲裁员", 3, "supplement"
            )]

        with patch.dict(app_config.AGENT_CONFIG, {"report_format_repair_enabled": True}):
            events = asyncio.run(collect())
        payloads = [json.loads(item["data"]) for item in events]
        tool_result = next(
            payload for item, payload in zip(events, payloads)
            if item["event"] == "tool_result"
        )
        self.assertEqual(tool_result["evidence_sources"][0]["evidence_id"], "RULE3")
        self.assertNotIn("risk_evidence_ids", tool_result)
        self.assertNotIn("enterprise_risk_levels", tool_result)
        self.assertNotIn("high_risk_evidence_ids", tool_result)
        self.assertNotIn("纳入对应风险等级的审查处置路径", requests[1]["messages"][-1]["content"])
        self.assertIn(
            '"enterprise_risk_level": "High"',
            requests[1]["messages"][-1]["content"],
        )
        self.assertEqual(
            [payload["stage"] for item, payload in zip(events, payloads)
             if item["event"] == "pipeline_stage"],
            [
                "tool_recording",
                "review_complete", "tool_recording",
            ],
        )
        final = next(
            payload for item, payload in zip(events, payloads)
            if item["event"] == "final_report"
        )
        self.assertEqual(final["raw_report"], report)
        self.assertEqual(len(requests), 2)

    def test_final_report_request_pins_enterprise_rule_evidence(self):
        tool_names = [
            "search_civil_code",
            "get_company_policy",
            "get_past_review_rules",
            "search_general_materials",
        ]
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                f"call_{name}", name, json.dumps({"query": f"查询 {name}"})
            )], finish_reason="tool_calls")]
            for name in tool_names
        ] + [[_chunk(tool_calls=[_tool_call_delta(
            "call_final",
            "submit_final_report",
            _final_report_arguments(),
        )], finish_reason="tool_calls")]])
        mapping = {
            name: (lambda query: json.dumps({"evidence": []}))
            for name in tool_names
        }
        mapping["get_company_policy"] = lambda query: json.dumps({
            "evidence": [
                {
                    "id": "KB_LONG",
                    "source_type": "enterprise_document",
                    "title": "企业审查偏好",
                    "source_location": "data/enterprise.md",
                    "text": "长篇企业文档内容。" * 90,
                },
                {
                    "id": "RULE3",
                    "source_type": "enterprise_rule",
                    "title": "争议管辖与仲裁机构",
                    "source_location": "data/rule_book.db · 规则 RULE3",
                    "text": "禁止由对方单方指定独任仲裁员在其办公室内仲裁。",
                },
            ],
        }, ensure_ascii=False)
        agent, requests = _agent(responses, mapping)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "仲裁条款", 1, "required-tools"
            )]

        with patch.dict(app_config.AGENT_CONFIG, {"report_format_repair_enabled": True}):
            events = asyncio.run(collect())
        forced_names = [
            request["tool_choice"]["function"]["name"]
            for request in requests[:len(tool_names)]
        ]
        self.assertEqual(forced_names, tool_names)
        final_event = next(
            (event for event in events if event["event"] == "final_report"),
            None,
        )
        self.assertIsNotNone(
            final_event,
            json.dumps([event["data"] for event in events[-3:]], ensure_ascii=False),
        )
        final = json.loads(final_event["data"])
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["turns"], 5)
        self.assertEqual(len(requests), 5)
        final_evidence_prompt = requests[-1]["messages"][-1]["content"]
        self.assertEqual(requests[-1]["messages"][-1]["role"], "user")
        self.assertIn("RULE3", final_evidence_prompt)
        self.assertIn("[[RULE:RULE3]]", final_evidence_prompt)
        self.assertIn("禁止由对方单方指定独任仲裁员", final_evidence_prompt)

    def test_rejects_final_report_until_each_read_tool_has_been_called(self):
        tool_names = [
            "search_civil_code",
            "get_company_policy",
            "get_past_review_rules",
            "search_general_materials",
        ]
        responses = iter([[
            _chunk(tool_calls=[_tool_call_delta(
                "premature_final",
                "submit_final_report",
                _final_report_arguments(_valid_report_payload(issue="过早报告")),
            )], finish_reason="tool_calls")
        ]] + [[_chunk(tool_calls=[_tool_call_delta(
            f"call_{name}", name, json.dumps({"query": f"查询 {name}"})
        )], finish_reason="tool_calls")] for name in tool_names] + [[_chunk(
            tool_calls=[_tool_call_delta(
                "call_final",
                "submit_final_report",
                _final_report_arguments(_valid_report_payload(issue="覆盖后报告")),
            )],
            finish_reason="tool_calls",
        )]])
        mapping = {
            name: (lambda query: json.dumps({"evidence": []}))
            for name in tool_names
        }
        agent, requests = _agent(responses, mapping)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "仲裁条款", 1, "early-final"
            )]

        events = asyncio.run(collect())
        payloads = [json.loads(item["data"]) for item in events]
        rejected_final = next(
            payload for item, payload in zip(events, payloads)
            if item["event"] == "tool_result"
            and payload.get("call_id") == "premature_final"
        )
        final = next(payload for payload in payloads if payload.get("status") == "success")
        self.assertFalse(rejected_final["success"])
        self.assertIn("尚未调用所有必需检索工具", rejected_final["error"])
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["turns"], 6)

    def test_invalid_final_schema_is_returned_as_tool_error_for_retry(self):
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                "bad_final",
                "submit_final_report",
                '{"report":" ","acknowledged_guardrails":[]}',
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "good_final",
                "submit_final_report",
                _final_report_arguments(),
            )], finish_reason="tool_calls")],
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 3, "retry"
            )]

        events = asyncio.run(collect())
        names = [item["event"] for item in events]
        self.assertEqual(names.count("tool_result"), 1)
        self.assertEqual(names[-2:], ["final_report", "done"])
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "bad_final")

    def test_invalid_guardrail_names_get_a_bounded_final_report_retry(self):
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                "bad_guardrails",
                "submit_final_report",
                json.dumps({
                    "report": _valid_report_payload(),
                    "acknowledged_guardrails": ["get_company_policy", "search_civil_code"],
                }),
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "corrected_final",
                "submit_final_report",
                json.dumps({
                    "report": _valid_report_payload(),
                    "acknowledged_guardrails": [],
                }),
            )], finish_reason="tool_calls")],
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 1, "guardrail-retry"
            )]

        events = asyncio.run(collect())
        payloads = [json.loads(item["data"]) for item in events]
        final = next(
            item for item in payloads
            if item.get("status") == "success"
        )
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["turns"], 2)
        self.assertEqual(requests[1]["tool_choice"], "required")
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[1]["tools"]],
            ["submit_final_report"],
        )
        feedback = requests[1]["messages"][-1]["content"]
        self.assertIn("不得填写工具名", feedback)

    def test_final_report_correction_retries_are_bounded(self):
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                f"invalid_final_{attempt}",
                "submit_final_report",
                '{"report":" ","acknowledged_guardrails":[]}',
            )], finish_reason="tool_calls")]
            for attempt in range(3)
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 1, "bounded-final-retries"
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(item["data"])
            for item in events
            if item["event"] == "final_report"
        )
        self.assertEqual(len(requests), 3)
        self.assertFalse(final["is_complete"])
        self.assertEqual(final["status"], "incomplete")
        self.assertEqual(final["turns"], 3)

    def test_truncated_structured_final_is_returned_as_tool_error_for_retry(self):
        responses = iter([
            [_chunk(
                tool_calls=[_tool_call_delta(
                    "truncated_final",
                    "submit_final_report",
                    '{"report":"语法完整但已被截断","acknowledged_guardrails":[]}',
                )],
                finish_reason="length",
            )],
            [_chunk(tool_calls=[_tool_call_delta(
                "complete_final",
                "submit_final_report",
                _final_report_arguments(),
            )], finish_reason="tool_calls")],
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 3, "truncated"
            )]

        events = asyncio.run(collect())
        names = [item["event"] for item in events]
        self.assertEqual(names.count("tool_result"), 1)
        self.assertEqual(names[-2:], ["final_report", "done"])
        feedback = requests[1]["messages"][-1]
        self.assertEqual(feedback["role"], "tool")
        self.assertEqual(feedback["tool_call_id"], "truncated_final")
        self.assertIn("输出长度上限", feedback["content"])

    def test_repeated_searches_force_final_report_tool(self):
        responses = iter([
            [_chunk(tool_calls=[_tool_call_delta(
                "search_1", "search_civil_code", '{"query":"重复查询"}'
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "search_2", "search_civil_code", '{"query":"重复查询"}'
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "search_3", "search_civil_code", '{"query":"重复查询"}'
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "final", "submit_final_report",
                _final_report_arguments()
            )], finish_reason="tool_calls")],
        ])
        executed_queries = []
        agent, requests = _agent(
            responses,
            {"search_civil_code": lambda query: executed_queries.append(query) or "{}"},
        )

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 5, "duplicate"
            )]

        events = asyncio.run(collect())
        self.assertEqual(len(executed_queries), 1)
        self.assertEqual(requests[3]["tool_choice"], "required")
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[3]["tools"]],
            ["submit_final_report"],
        )
        self.assertEqual(events[-2]["event"], "final_report")

    def test_prefaced_plain_text_final_finishes_without_repeat_submission(self):
        report_payload = _valid_report_payload()
        report_json = _canonical_report_json(report_payload)
        responses = iter([
            [_chunk(
                content="最终报告如下：\n" + json.dumps(report_payload, ensure_ascii=False),
                finish_reason="stop",
            )],
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 3, "plain-text-final"
            )]

        events = asyncio.run(collect())
        self.assertEqual(len(requests), 1)
        self.assertEqual([item["event"] for item in events][-2:], ["final_report", "done"])
        final = json.loads(events[-2]["data"])
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["raw_report"], report_json)

    def test_iteration_limit_never_returns_success(self):
        responses = iter([[_chunk(content="请继续", finish_reason="stop")]])
        agent, _ = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 1, "limit"
            )]

        events = asyncio.run(collect())
        final = next(
            json.loads(item["data"])
            for item in events if item["event"] == "final_report"
        )
        self.assertFalse(final["is_complete"])
        self.assertEqual(final["status"], "incomplete")
        self.assertEqual(events[-1]["event"], "error")


if __name__ == "__main__":
    unittest.main()