import json
import unittest

from services.review_execution import execute_review_unit


class ReviewExecutionTests(unittest.TestCase):
    def test_success_collects_one_canonical_result_for_any_caller(self):
        raw_report = json.dumps({
            "reviews": [{
                "clause_topic": "第五条 验收",
                "legal_basis": "未检索到直接依据",
                "issue": "验收标准应进一步明确。",
                "suggested_revision": "补充验收期限。",
            }],
        }, ensure_ascii=False)
        events = [
            {"event": "token", "data": {"token": "模型文本"}},
            {"event": "report_token", "data": {"token": "报告片段"}},
            {
                "event": "tool_result",
                "data": {
                    "observation": '{"evidence":[{"id":"EV1","tag":"法"}]}',
                },
            },
            {
                "event": "final_report",
                "data": {"raw_report": raw_report, "is_complete": True},
            },
            {"event": "done", "data": {}},
        ]
        observed = []

        result = execute_review_unit(
            "http://gateway",
            "合同正文",
            4,
            "buyer",
            on_event=lambda event, data: observed.append(event),
            stream_factory=lambda *args, **kwargs: iter(events),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["report"], raw_report)
        self.assertNotIn("model_text", result)
        self.assertNotIn("model_report", result)
        self.assertEqual(result["evidence_records"]["EV1"]["source_type"], "law")
        self.assertEqual(observed, [item["event"] for item in events])

    def test_rejects_unstructured_or_empty_final_report(self):
        for raw_report in (
            "模型输出的中间分析文本",
            '{"reviews":[]}',
        ):
            with self.subTest(raw_report=raw_report):
                result = execute_review_unit(
                    "http://gateway",
                    "合同正文",
                    4,
                    "neutral",
                    stream_factory=lambda *args, **kwargs: iter([{
                        "event": "final_report",
                        "data": {"raw_report": raw_report, "is_complete": True},
                    }]),
                )

                self.assertFalse(result["success"])
                self.assertFalse(result["report"])
                self.assertIn("结构化审查条目", result["error"])

    def test_incomplete_and_error_streams_share_failure_semantics(self):
        for events in (
            [{"event": "done", "data": {}}],
            [{"event": "error", "data": {"error": "gateway unavailable"}}],
            [{
                "event": "final_report",
                "data": {"raw_report": "partial", "is_complete": False},
            }],
        ):
            with self.subTest(events=events):
                result = execute_review_unit(
                    "http://gateway",
                    "合同正文",
                    4,
                    "neutral",
                    stream_factory=lambda *args, **kwargs: iter(events),
                )
                self.assertFalse(result["success"])
                self.assertFalse(result["report"])
                self.assertTrue(result["error"])


if __name__ == "__main__":
    unittest.main()
