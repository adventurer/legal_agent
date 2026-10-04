import json
import unittest
from types import SimpleNamespace

from services.contract_rewriter import rewrite_selected_clauses


class _FakeCompletions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responses[min(
            len(self.requests) - 1,
            len(self.responses) - 1,
        )]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


def _agent_with_responses(*responses):
    completions = _FakeCompletions(*responses)
    agent = SimpleNamespace(
        model_name="test-model",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
    )
    return agent, completions


def _model_response(index, revised_text):
    return json.dumps(
        {
            "revised_clauses": [{
                "index": index,
                "revised_text": revised_text,
                "change_reason": "修订条款",
            }]
        },
        ensure_ascii=False,
    )


class ContractRewriterTests(unittest.TestCase):
    def setUp(self):
        self.original_text = (
            "8.1 友好协商：凡因本合同引起的争议，双方应首先友好协商解决。\n"
            "8.2 仲裁管辖约定：如协商不成，双方同意提交仲裁。"
        )
        self.clauses = [{
            "index": 8,
            "title": "第八条 争议解决与仲裁",
            "content": self.original_text,
            "raw_text": f"第八条 争议解决与仲裁\n{self.original_text}",
        }]

    def test_preserves_every_numbered_subclause(self):
        revised_text = (
            "8.1 友好协商：双方应先通过高级管理人员协商解决争议。\n"
            "8.2 仲裁管辖约定：协商未果时，双方同意将争议交由仲裁处理。"
        )
        agent, completions = _agent_with_responses(
            _model_response(8, revised_text)
        )

        result = rewrite_selected_clauses(
            agent,
            self.clauses,
            "第八条 争议解决与仲裁\n修改建议：明确仲裁机构。",
            [8],
        )

        self.assertIn("8.1 友好协商", result["contract_text"])
        self.assertIn("8.2 仲裁管辖约定", result["contract_text"])
        self.assertIn("全部子条款", completions.requests[0]["messages"][0]["content"])
        self.assertIn('"index":8', completions.requests[0]["messages"][0]["content"])

    def test_retries_when_model_omits_subclause_and_accepts_complete_retry(self):
        incomplete_text = "8.1 友好协商：双方应先协商解决争议。"
        complete_text = (
            "8.1 友好协商：双方应先通过高级管理人员协商解决争议。\n"
            "8.2 仲裁管辖约定：协商未果时，双方同意将争议交由仲裁处理。"
        )
        agent, completions = _agent_with_responses(
            _model_response(8, incomplete_text),
            _model_response(8, complete_text),
        )

        result = rewrite_selected_clauses(agent, self.clauses, "", [8])

        self.assertEqual(len(completions.requests), 2)
        self.assertIn(
            "上一次修订遗漏或重排了子条款编号",
            completions.requests[1]["messages"][0]["content"],
        )
        self.assertIn("8.2 仲裁管辖约定", result["contract_text"])

    def test_rejects_incomplete_clause_after_retry(self):
        incomplete_text = "8.1 友好协商：双方应先协商解决争议。"
        agent, completions = _agent_with_responses(
            _model_response(8, incomplete_text),
            _model_response(8, incomplete_text),
        )

        with self.assertRaisesRegex(ValueError, "两次修订均遗漏或重排"):
            rewrite_selected_clauses(agent, self.clauses, "", [8])

        self.assertEqual(len(completions.requests), 2)


if __name__ == "__main__":
    unittest.main()
