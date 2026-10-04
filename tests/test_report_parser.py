import unittest

from core.prompts import RISK_LEVEL_REPORT_LEGEND
from services.report_parser import (
    normalize_report_structure,
    is_final_report,
    parse_structured_report,
    render_report_article,
    restore_clause_numbers,
    validate_report_structure,
)


class ReportParserTests(unittest.TestCase):
    def test_restores_missing_article_and_subclause_numbers_from_contract(self):
        report = parse_structured_report(
            '{"reviews":['
            '{"clause_topic":"争议解决与仲裁","risk_level":"High",'
            '"legal_basis":"无直接依据","issue":"争议机制失衡。",'
            '"suggested_revision":"明确仲裁机构。"},'
            '{"clause_topic":"友好协商","risk_level":"Low",'
            '"legal_basis":"无直接依据","issue":"程序不清晰。",'
            '"suggested_revision":"补充协商期限。"},'
            '{"clause_topic":"仲裁管辖约定","risk_level":"High",'
            '"legal_basis":"无直接依据","issue":"管辖约定不明确。",'
            '"suggested_revision":"明确仲裁地点。"}'
            ']}'
        )
        self.assertIsNotNone(report)
        contract_text = (
            "第八条 争议解决与仲裁\n"
            "8.1 友好协商：双方先行协商。\n"
            "8.2 仲裁管辖约定：双方提交仲裁。"
        )

        restored = restore_clause_numbers(report, contract_text)

        self.assertEqual(
            [item.clause_topic for item in restored.reviews],
            [
                "第八条 争议解决与仲裁",
                "8.1 友好协商",
                "8.2 仲裁管辖约定",
            ],
        )

    def test_corrects_report_sequence_number_and_does_not_guess_ambiguous_title(self):
        report = parse_structured_report(
            '{"reviews":['
            '{"clause_topic":"1. 付款安排","risk_level":"Low",'
            '"legal_basis":"无直接依据","issue":"条款需明确。",'
            '"suggested_revision":"明确付款期限。"},'
            '{"clause_topic":"重复标题","risk_level":"Low",'
            '"legal_basis":"无直接依据","issue":"条款需明确。",'
            '"suggested_revision":"进一步明确。"}'
            ']}'
        )
        self.assertIsNotNone(report)
        contract_text = (
            "第八条 付款安排\n约定付款期限。\n"
            "第九条 重复标题\n约定甲方责任。\n"
            "第十条 重复标题\n约定乙方责任。"
        )

        restored = restore_clause_numbers(report, contract_text)

        self.assertEqual(restored.reviews[0].clause_topic, "第八条 付款安排")
        self.assertEqual(restored.reviews[1].clause_topic, "重复标题")

    def test_renders_json_report_as_readable_markdown_article(self):
        report = parse_structured_report(
            '{"reviews":[{"clause_topic":"第五条 验收","risk_level":"High",'
            '"legal_basis":"依据 [[EVIDENCE:EV1]]","issue":"验收机制不明确。",'
            '"suggested_revision":"补充验收期限。"}]}'
        )

        article = render_report_article(report)

        self.assertIn("# 合同审查报告", article)
        self.assertIn("## 1. 第五条 验收", article)
        self.assertIn("**风险等级**: 高风险", article)
        self.assertIn("**风险剖析**: 验收机制不明确。", article)
        self.assertIn("[[EVIDENCE:EV1]]", article)

    def test_parses_json_report_inside_json_code_fence(self):
        raw_report = (
            '```json\n{"reviews":[{"clause_topic":"第五条 验收",'
            '"risk_level":"Low","legal_basis":"无直接依据",'
            '"issue":"期限不明确。","suggested_revision":"补充期限。"}]}\n```'
        )
        report = parse_structured_report(raw_report)

        self.assertIsNotNone(report)
        self.assertEqual(report.reviews[0].clause_topic, "第五条 验收")
        self.assertTrue(is_final_report(raw_report))

    def test_normalizes_heading_fields_and_common_affected_party_typo(self):
        raw_report = """# 第五条 服务保障

## 5.1 质保周期
### 风险类型: 法律合规
### 风险等级: 低风险
### 企业内部风险等级: 
### 法律效力: 有直接依据
### 商业后果: 保障设备质量
### 救济成本: 低
### 影响受方: 双方
### 结论置信度: 高
### 法律/合规依据: 民法典合同编第619条
### 企业知识库依据: 
### 风险剖析: 约定提供质保服务。
### 修改建议: 明确质保期限。
"""

        normalized = normalize_report_structure(raw_report)

        self.assertIsNotNone(normalized)
        self.assertIn("### 风险等级提示", normalized)
        self.assertIn("- **受影响方**: 双方", normalized)
        self.assertIn("- **企业内部风险等级：** 未检索到企业内部风险等级", normalized)
        self.assertIn(
            "- **企业知识库依据**: 未检索到相关企业规则或知识库依据",
            normalized,
        )
        self.assertNotIn("初稿未提供", normalized)
        self.assertEqual(validate_report_structure(normalized), [])
        parsed = parse_structured_report(normalized)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.reviews[0].affected_party, "双方")
        self.assertEqual(
            parsed.reviews[0].enterprise_risk_level,
            "未检索到企业内部风险等级",
        )
        self.assertEqual(
            parsed.reviews[0].enterprise_basis,
            "未检索到相关企业规则或知识库依据",
        )

    def test_normalization_preserves_populated_enterprise_evidence_fields(self):
        raw_report = f"""{RISK_LEVEL_REPORT_LEGEND}

    ### 第三条 资料使用
- **风险类型**: 商业
- **风险等级**: 中风险
- **企业内部风险等级**: High
- **法律效力**: 未检索到直接依据
- **商业后果**: 需复核企业流程
- **救济成本**: 中
- **受影响方**: 甲方
- **结论置信度**: 中
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: reference_docs/policy.md · 第 4 页 [[KB:EV12]]
- **风险剖析**: 流程要求未写入合同。
- **修改建议**: 补充流程约定。
"""

        normalized = normalize_report_structure(raw_report)

        self.assertIn("- **企业内部风险等级：** High", normalized)
        self.assertIn(
            "- **企业知识库依据**: reference_docs/policy.md · 第 4 页 [[KB:EV12]]",
            normalized,
        )

    def test_normalization_keeps_fields_before_embedded_contract_heading(self):
        raw_report = f"""{RISK_LEVEL_REPORT_LEGEND}

**风险类型**: 法律合规
**风险等级**: 中风险
**企业内部风险等级**: High
**法律效力**: 有直接依据的判断
**商业后果**: 尾款支付周期较长
**救济成本**: 中
**受影响方**: 乙方
**结论置信度**: 中
**法律/合规依据**: [[RULE:RULE2]]
**企业知识库依据**: [[KB:EVC7FC2169C3B9]]
**风险剖析**: 尾款支付周期可能影响乙方资金流动性。
**修改建议**: 将尾款支付周期限定为30个工作日。

# 第二条 验收标准与付款周期

合同原文内容。
"""

        normalized = normalize_report_structure(raw_report)

        self.assertIn("- **风险类型**: 法律合规", normalized)
        self.assertIn("- **风险等级**: 中风险", normalized)
        self.assertIn("- **企业内部风险等级：** High", normalized)
        self.assertIn("- **受影响方**: 乙方", normalized)
        self.assertIn("尾款支付周期可能影响乙方资金流动性。", normalized)
        self.assertIn("将尾款支付周期限定为30个工作日。", normalized)
        self.assertNotIn("初稿未提供", normalized)

    def test_normalization_applies_validated_supplement_over_no_match_defaults(self):
        raw_report = """### 第五条 服务保障
- **风险等级**: 中风险
- **法律/合规依据**: 未检索到直接依据
- **风险剖析**: 需核对服务保障流程。
- **修改建议**: 明确服务流程。
"""

        normalized = normalize_report_structure(
            raw_report,
            evidence_supplements={
                "第五条 服务保障": {
                    "enterprise_risk_level": "High",
                    "enterprise_basis": "data/rule_book.db · 规则 RULE3 [[RULE:RULE3]]",
                },
            },
        )

        self.assertIn("- **企业内部风险等级：** High", normalized)
        self.assertIn(
            "- **企业知识库依据**: data/rule_book.db · 规则 RULE3 [[RULE:RULE3]]",
            normalized,
        )

    def test_parses_bold_enterprise_risk_label_with_chinese_colon_inside(self):
        raw_report = f"""{RISK_LEVEL_REPORT_LEGEND}

    ### 第三条 资料使用
- **风险类型**: 商业
- **风险等级**: 中风险
- **企业内部风险等级：** 未检索到企业内部风险等级
- **法律效力**: 未检索到直接依据
- **商业后果**: 暂无直接影响资料。
- **救济成本**: 低
- **受影响方**: 双方
- **结论置信度**: 低
- **法律/合规依据**: 未检索到直接依据
- **企业知识库依据**: 未检索到相关企业规则或知识库依据
- **风险剖析**: 知识库无相关规则。
- **修改建议**: 补充企业规则资料。
"""

        self.assertEqual(validate_report_structure(raw_report), [])
        parsed = parse_structured_report(raw_report)
        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed.reviews[0].enterprise_risk_level,
            "未检索到企业内部风险等级",
        )


if __name__ == "__main__":
    unittest.main()