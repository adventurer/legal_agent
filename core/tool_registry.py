"""合同审查 Agent 的知识工具注册。"""

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from core.tools import (
    get_company_policy,
    get_past_review_rules,
    search_civil_code,
    search_general_materials,
)
from core.tools.common import search_tagged_documents


class ReviewToolRegistry:
    """初始化并保存法规库、企业规则库工具。"""

    def __init__(
        self,
        docs_dir: str,
        db_path: str,
        pdf_kb_class: Optional[Any],
        rule_book_class: Optional[Any],
    ):
        self.docs_dir = docs_dir
        self.db_path = db_path
        self.pdf_kb_class = pdf_kb_class
        self.rule_book_class = rule_book_class
        self.knowledge_base = None
        self.rule_book = None
        self.unavailable_tools = set()

    def _unavailable(self, tool_name: str):
        def raise_unavailable(_query: str) -> str:
            raise RuntimeError(f"工具 {tool_name} 不可用：知识库目录或服务未就绪。")
        return raise_unavailable

    def _search_tagged_documents(self, query: str, tag: str, source_type: str) -> List[Dict[str, Any]]:
        return search_tagged_documents(
            self.knowledge_base,
            self.docs_dir,
            query,
            tag,
            source_type,
        )

    def _search_enterprise_rules(self, query: str) -> List[Dict[str, Any]]:
        if not self.rule_book:
            return []
        return self.rule_book.search_rule_evidence(query)

    def build(self) -> Dict[str, Callable[[str], str]]:
        tools: Dict[str, Callable[[str], str]] = {}
        self.unavailable_tools.clear()
        if self.pdf_kb_class and os.path.exists(self.docs_dir):
            print(f"[*] 正在从配置目录载入权威 PDF 参考库: {self.docs_dir}", flush=True)
            self.knowledge_base = self.pdf_kb_class(self.docs_dir)
            for name, tag in (("search_civil_code", "法"),):
                if any(page.get("tag") == tag for page in self.knowledge_base.pages):
                    tools[name] = partial(
                        search_civil_code,
                        search_documents=self._search_tagged_documents,
                    )
                else:
                    print(f"[!] 参考库中没有 [{tag}] 类有效页面，{name} 将报告不可用。", flush=True)
                    tools[name] = self._unavailable(name)
                    self.unavailable_tools.add(name)
            for tool_name, tag, source_type in (
                ("get_company_policy", "合规", "enterprise_document"),
                ("search_general_materials", "通用", "general_document"),
            ):
                has_pages = any(
                    page.get("tag") == tag for page in self.knowledge_base.pages
                )
                if tool_name == "get_company_policy" and self.rule_book:
                    has_pages = True
                if has_pages:
                    tools[tool_name] = (
                        partial(
                            get_company_policy,
                            search_documents=self._search_tagged_documents,
                            search_rules=self._search_enterprise_rules,
                        )
                        if tool_name == "get_company_policy"
                        else partial(
                            search_general_materials,
                            search_documents=self._search_tagged_documents,
                        )
                    )
                    self.unavailable_tools.discard(tool_name)
                else:
                    tools[tool_name] = self._unavailable(tool_name)
                    self.unavailable_tools.add(tool_name)
        else:
            print("[!] PDF 参考库目录不存在或解析组件缺失，法规工具将报告不可用。", flush=True)
            tools["search_civil_code"] = self._unavailable("search_civil_code")
            self.unavailable_tools.add("search_civil_code")
            tools["get_company_policy"] = self._unavailable("get_company_policy")
            tools["search_general_materials"] = self._unavailable("search_general_materials")
            self.unavailable_tools.update({"get_company_policy", "search_general_materials"})

        if self.rule_book_class:
            print(f"[*] 正在连接企业自编法典数据库: {self.db_path}", flush=True)
            self.rule_book = self.rule_book_class(db_path=self.db_path)
            tools["get_past_review_rules"] = partial(
                get_past_review_rules,
                search_rules=self._search_enterprise_rules,
            )
        else:
            print("[!] 未检测到 EnterpriseRuleBook 服务，自编法典工具将报告不可用。", flush=True)
            tools["get_past_review_rules"] = self._unavailable("get_past_review_rules")
            self.unavailable_tools.add("get_past_review_rules")

        has_company_documents = bool(
            self.knowledge_base
            and any(page.get("tag") == "合规" for page in self.knowledge_base.pages)
        )
        if has_company_documents or self.rule_book:
            tools["get_company_policy"] = partial(
                get_company_policy,
                search_documents=self._search_tagged_documents,
                search_rules=self._search_enterprise_rules,
            )
            self.unavailable_tools.discard("get_company_policy")
        else:
            tools["get_company_policy"] = self._unavailable("get_company_policy")
            self.unavailable_tools.add("get_company_policy")
        return tools
