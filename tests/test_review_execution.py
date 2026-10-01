import unittest

from services.review_execution import execute_review_unit


class ReviewExecutionTests(unittest.TestCase):
    def test_success_collects_one_canonical_result_for_any_caller(self):
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
                "data": {"raw_report": "最终报告原文", "is_complete": True},
            },
            {"event": "done", "data": {}},
        ]
        observed = []

        result = execute_review_unit(
            "http://gateway",
            "合同正文",
            4,
            "run-1",
            "buyer",
            on_event=lambda event, data: observed.append(event),
            stream_factory=lambda *args, **kwargs: iter(events),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["report"], "最终报告原文")
        self.assertEqual(result["model_text"], "模型文本")
        self.assertEqual(result["model_report"], "最终报告原文")
        self.assertEqual(result["evidence_records"]["EV1"]["source_type"], "law")
        self.assertEqual(observed, [item["event"] for item in events])

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
                    "run-1",
                    "neutral",
                    stream_factory=lambda *args, **kwargs: iter(events),
                )
                self.assertFalse(result["success"])
                self.assertFalse(result["report"])
                self.assertTrue(result["error"])


if __name__ == "__main__":
    unittest.main()
