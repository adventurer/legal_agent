import unittest

from core.prompts import RISK_LEVEL_REPORT_LEGEND
from services.report_parser import (
    normalize_report_structure,
    parse_structured_report,
    validate_report_structure,
)


class ReportParserTests(unittest.TestCase):
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