"""Contract tests for the native-tool-call LLM interaction layer."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from gateway.llm_interaction.context_manager import ContextWindowManager
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
    REPORT_OUTPUT_GUIDANCE,
    REVIEW_ANALYSIS_GUIDANCE,
    RISK_LEVEL_REPORT_LEGEND,
)
from services.pdf_kb_search import search_pages


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
        self.assertIn("必须为空数组", schemas[-1]["function"]["description"])

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
            arguments=json.dumps({
                "report": "报告正文",
                "acknowledged_guardrails": [],
            }),
        )
        with self.assertRaises(FinalReportValidationError):
            validate_final_report(call, ["unlimited_liability"])

class ContextAndGuardrailTests(unittest.TestCase):
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
    def test_native_prompt_includes_shared_review_requirements(self):
        messages = build_review_messages("第一条 合同期限", "buyer", [])
        system_prompt = messages[0]["content"]

        self.assertIn(REVIEW_ANALYSIS_GUIDANCE, system_prompt)
        self.assertIn(REPORT_OUTPUT_GUIDANCE, system_prompt)
        self.assertIn("逾期 30 日", system_prompt)
        self.assertIn("按规则的风险等级、审查标准和禁止情形逐项核对", system_prompt)
        self.assertIn("[[EVIDENCE:EV编号]]", system_prompt)
        self.assertIn("submit_final_report", system_prompt)
        self.assertIn("不得填写工具名", system_prompt)
        self.assertNotIn("工具轮只输出 Thought 和一行 Action", system_prompt)
        self.assertNotIn("最终轮以 Thought 和 Final: 开始", system_prompt)


class ReActLoopTests(unittest.TestCase):
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

    def test_finalizer_always_validates_a_structurally_valid_report(self):
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
        agent, requests = _agent(iter([
            [_chunk(content=report, finish_reason="stop")],
        ]))
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

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertEqual(len(requests), 1)
        self.assertIn("完整 Markdown", requests[0]["messages"][0]["content"])
        self.assertEqual(final["status"], "success")
        self.assertIn("### 第五条 服务保障", final["raw_report"])

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
        agent, requests = _agent(iter([
            [_chunk(content=repaired_report, finish_reason="stop")],
        ]))
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

        events = asyncio.run(collect())
        final = next(
            json.loads(event["data"])
            for event in events
            if event["event"] == "final_report"
        )
        self.assertNotIn("初稿未提供", final["raw_report"])
        self.assertIn("- **风险类型**: 商业", final["raw_report"])
        self.assertIn("初稿未提供", requests[0]["messages"][0]["content"])

    def test_native_tool_call_returns_role_tool_then_structured_final(self):
        report = "建议明确付款期限。"
        report_arguments = json.dumps({
            "report": report,
            "acknowledged_guardrails": [],
        }, ensure_ascii=False)
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
        self.assertIn(report, model_tool_calls[1]["arguments"])
        self.assertEqual(names[-2:], ["final_report", "done"])
        self.assertEqual(names.count("tool_start"), 1)
        self.assertEqual(names.count("tool_result"), 1)
        self.assertEqual(
            "".join(item["token"] for item in payloads if item.get("token") and "report_token" in names),
            report,
        )
        report_tokens = [
            json.loads(item["data"])["token"]
            for item in events if item["event"] == "report_token"
        ]
        self.assertEqual("".join(report_tokens), report)
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

    def test_enterprise_risk_evidence_handles_all_levels(self):
        evidence = {
            evidence_id: {
                "source_type": "enterprise_rule",
                "enterprise_risk_level": risk_level,
            }
            for evidence_id, risk_level in [
                ("RULE_HIGH", "High"),
                ("RULE_MEDIUM", "Medium"),
                ("RULE_LOW", "Low"),
                ("RULE_NOTICE", "Notice"),
            ]
        }

        self.assertEqual(
            ReactLoop._enterprise_risk_evidence(evidence),
            {
                "RULE_HIGH": "High",
                "RULE_MEDIUM": "Medium",
                "RULE_LOW": "Low",
                "RULE_NOTICE": "Notice",
            },
        )

    def test_finalization_captures_and_supplements_enterprise_evidence(self):
        report = "### 8.2 仲裁管辖约定\n- **风险等级**: 中风险"
        formatted_report = f"""{RISK_LEVEL_REPORT_LEGEND}

    ### 8.2 仲裁管辖约定
- **风险类型**: 商业
- **风险等级**: 中风险
- **企业内部风险等级**: Low
- **法律效力**: 未检索到直接依据
- **商业后果**: 可能增加争议处理成本。
- **救济成本**: 中
- **受影响方**: 双方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: stale.db · 规则 OLD [[RULE:OLD]]
- **风险剖析**: 仲裁员选任方式需核实。
- **修改建议**: 明确仲裁员选任方式。
"""
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
                    "report": report,
                    "acknowledged_guardrails": [],
                }, ensure_ascii=False),
            )], finish_reason="tool_calls")],
            [_chunk(content=formatted_report, finish_reason="stop")],
            [_chunk(content=json.dumps({
                "supplements": [{
                    "clause_topic": "8.2 仲裁管辖约定",
                    "evidence_ids": ["RULE3"],
                    "raise_to_high_risk": True,
                }],
            }, ensure_ascii=False), finish_reason="stop")],
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

        events = asyncio.run(collect())
        payloads = [json.loads(item["data"]) for item in events]
        tool_result = next(
            payload for item, payload in zip(events, payloads)
            if item["event"] == "tool_result"
        )
        self.assertEqual(tool_result["evidence_sources"][0]["evidence_id"], "RULE3")
        self.assertEqual(tool_result["risk_evidence_ids"], ["RULE3"])
        self.assertEqual(tool_result["enterprise_risk_levels"], {"RULE3": "High"})
        self.assertEqual(tool_result["high_risk_evidence_ids"], ["RULE3"])
        self.assertIn(
            "纳入对应风险等级的审查处置路径",
            requests[1]["messages"][-1]["content"],
        )
        self.assertIn("High", requests[1]["messages"][-1]["content"])
        self.assertEqual(
            [payload["stage"] for item, payload in zip(events, payloads)
             if item["event"] == "pipeline_stage"],
            [
                "tool_recording",
                "format_validation", "format_validation",
                "report_reconciliation", "report_reconciliation",
                "review_complete", "tool_recording",
            ],
        )
        final = next(
            payload for item, payload in zip(events, payloads)
            if item["event"] == "final_report"
        )
        self.assertIn("High", final["raw_report"])
        self.assertIn("- **风险等级**: 高风险", final["raw_report"])
        self.assertIn("data/rule_book.db · 规则 RULE3 [[RULE:RULE3]]", final["raw_report"])
        supplement_input = json.loads(requests[3]["messages"][1]["content"])
        self.assertEqual(supplement_input["tool_call_records"][0]["tool"], "get_company_policy")
        self.assertEqual(supplement_input["tool_call_records"][0]["evidence_ids"], ["RULE3"])
        self.assertTrue(supplement_input["tool_call_records"][0]["success"])
        self.assertEqual(len(requests), 4)

    def test_requires_each_available_read_tool_before_final_report(self):
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
            '{"report":"覆盖全部检索源","acknowledged_guardrails":[]}',
        )], finish_reason="tool_calls")]])
        mapping = {
            name: (lambda query: json.dumps({"evidence": []}))
            for name in tool_names
        }
        agent, requests = _agent(responses, mapping)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "仲裁条款", 1, "required-tools"
            )]

        events = asyncio.run(collect())
        forced_names = [
            request["tool_choice"]["function"]["name"]
            for request in requests[:len(tool_names)]
        ]
        self.assertEqual(forced_names, tool_names)
        self.assertTrue(json.loads(events[-2]["data"])["is_complete"])
        self.assertEqual(json.loads(events[-2]["data"])["turns"], 5)
        self.assertEqual(len(requests), 6)
        self.assertIn("格式修复器", requests[-1]["messages"][0]["content"])

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
                '{"report":"过早报告","acknowledged_guardrails":[]}',
            )], finish_reason="tool_calls")
        ]] + [[_chunk(tool_calls=[_tool_call_delta(
            f"call_{name}", name, json.dumps({"query": f"查询 {name}"})
        )], finish_reason="tool_calls")] for name in tool_names] + [[_chunk(
            tool_calls=[_tool_call_delta(
                "call_final",
                "submit_final_report",
                '{"report":"覆盖后报告","acknowledged_guardrails":[]}',
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
                '{"report":"有效报告","acknowledged_guardrails":[]}',
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
                    "report": "审查报告",
                    "acknowledged_guardrails": ["get_company_policy", "search_civil_code"],
                }),
            )], finish_reason="tool_calls")],
            [_chunk(tool_calls=[_tool_call_delta(
                "corrected_final",
                "submit_final_report",
                json.dumps({
                    "report": "审查报告",
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
                '{"report":"补全后的有效报告","acknowledged_guardrails":[]}',
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
                '{"report":"已基于现有资料完成","acknowledged_guardrails":[]}'
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

    def test_plain_text_final_forces_structured_submission(self):
        responses = iter([
            [_chunk(content="### 审查结论\n建议明确仲裁条款。", finish_reason="stop")],
            [_chunk(tool_calls=[_tool_call_delta(
                "final",
                "submit_final_report",
                json.dumps({
                    "report": "### 审查结论\n建议明确仲裁条款。",
                    "acknowledged_guardrails": [],
                }, ensure_ascii=False),
            )], finish_reason="tool_calls")],
        ])
        agent, requests = _agent(responses)

        async def collect():
            return [item async for item in ReactLoop(agent).stream(
                "合同正文", 3, "plain-text-final"
            )]

        events = asyncio.run(collect())
        self.assertEqual(requests[1]["tool_choice"], "required")
        self.assertEqual(
            [tool["function"]["name"] for tool in requests[1]["tools"]],
            ["submit_final_report"],
        )
        self.assertEqual([item["event"] for item in events][-2:], ["final_report", "done"])
        final = json.loads(events[-2]["data"])
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["raw_report"], "### 审查结论\n建议明确仲裁条款。")

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