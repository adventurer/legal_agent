"""Tests for the independently registered read-only review tools."""

import json
import unittest
from pathlib import Path

from core.tool_registry import ReviewToolRegistry


class FakeKnowledgeBase:
    pages = [
        {"tag": "法"},
        {"tag": "合规"},
        {"tag": "通用"},
    ]

    def __init__(self, docs_dir):
        self.docs_dir = docs_dir

    def search_keyword(self, query, filter_tag=None):
        return json.dumps({
            "evidence": [{
                "id": filter_tag,
                "doc_name": f"{filter_tag}.md",
                "page_start": 1,
            }],
        })


class FakeRuleBook:
    def __init__(self, db_path):
        self.db_path = db_path

    def search_rule_evidence(self, query):
        return [{"id": "RULE1", "source_type": "enterprise_rule"}]


class ReviewToolRegistryTests(unittest.TestCase):
    def test_registry_binds_each_tool_to_its_own_evidence_source(self):
        registry = ReviewToolRegistry(
            docs_dir=str(Path(__file__).parent),
            db_path="fake-rules.db",
            pdf_kb_class=FakeKnowledgeBase,
            rule_book_class=FakeRuleBook,
        )
        tools = registry.build()
        query = "仲裁协议效力"

        law = json.loads(tools["search_civil_code"](query))
        company = json.loads(tools["get_company_policy"](query))
        past_rules = json.loads(tools["get_past_review_rules"](query))
        general = json.loads(tools["search_general_materials"](query))

        self.assertEqual(law["evidence"][0]["source_type"], "law")
        self.assertEqual(law["evidence"][0]["source_location"].split(" · ")[0], "tests/法.md")
        self.assertEqual(
            {item["source_type"] for item in company["evidence"]},
            {"enterprise_document", "enterprise_rule"},
        )
        self.assertEqual(past_rules["evidence"][0]["id"], "RULE1")
        self.assertEqual(general["evidence"][0]["source_type"], "general_document")
        self.assertFalse(registry.unavailable_tools)


if __name__ == "__main__":
    unittest.main()