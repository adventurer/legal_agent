"""合同审查 Agent 的知识工具注册。"""

import os
from typing import Any, Callable, Dict, Optional


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

    def build(self) -> Dict[str, Callable[[str], str]]:
        tools: Dict[str, Callable[[str], str]] = {}
        if self.pdf_kb_class and os.path.exists(self.docs_dir):
            print(f"[*] 正在从配置目录载入权威 PDF 参考库: {self.docs_dir}", flush=True)
            self.knowledge_base = self.pdf_kb_class(self.docs_dir)
            for name, tag in (
                ("search_civil_code", "法"),
                ("get_company_policy", "合规"),
            ):
                if any(page.get("tag") == tag for page in self.knowledge_base.pages):
                    tools[name] = lambda query, selected_tag=tag: self.knowledge_base.search_keyword(
                        query, filter_tag=selected_tag
                    )
                else:
                    print(f"[!] 参考库中没有 [{tag}] 类有效页面，{name} 将报告不可用。", flush=True)
                    tools[name] = self._unavailable(name)
                    self.unavailable_tools.add(name)
        else:
            print("[!] PDF 参考库目录不存在或解析组件缺失，法规工具将报告不可用。", flush=True)
            for name in ("search_civil_code", "get_company_policy"):
                tools[name] = self._unavailable(name)
                self.unavailable_tools.add(name)

        if self.rule_book_class:
            print(f"[*] 正在连接企业自编法典数据库: {self.db_path}", flush=True)
            self.rule_book = self.rule_book_class(db_path=self.db_path)
            tools["get_past_review_rules"] = lambda query: self.rule_book.search_rules(query)
        else:
            print("[!] 未检测到 EnterpriseRuleBook 服务，自编法典工具将报告不可用。", flush=True)
            tools["get_past_review_rules"] = self._unavailable("get_past_review_rules")
            self.unavailable_tools.add("get_past_review_rules")
        return tools
