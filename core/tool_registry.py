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

    def build(self) -> Dict[str, Callable[[str], str]]:
        tools: Dict[str, Callable[[str], str]] = {}
        if self.pdf_kb_class and os.path.exists(self.docs_dir):
            print(f"[*] 正在从配置目录载入权威 PDF 参考库: {self.docs_dir}", flush=True)
            self.knowledge_base = self.pdf_kb_class(self.docs_dir)
            tools["search_civil_code"] = lambda query: self.knowledge_base.search_keyword(
                query, filter_tag="法"
            )
            tools["get_company_policy"] = lambda query: self.knowledge_base.search_keyword(
                query, filter_tag="合规"
            )
        else:
            print("[!] PDF 参考库目录不存在或解析组件缺失，启用法规占位桩。", flush=True)
            tools["search_civil_code"] = lambda query: f"[模拟] 查阅法规关于: {query}"
            tools["get_company_policy"] = lambda query: f"[模拟] 查阅政策关于: {query}"

        if self.rule_book_class:
            print(f"[*] 正在连接企业自编法典数据库: {self.db_path}", flush=True)
            self.rule_book = self.rule_book_class(db_path=self.db_path)
            tools["get_past_review_rules"] = lambda query: self.rule_book.search_rules(query)
        else:
            print("[!] 未检测到 EnterpriseRuleBook 服务，启用自编法典占位桩。", flush=True)
            tools["get_past_review_rules"] = (
                lambda query: f"《企业自编法典》中暂无针对【{query}】的特殊审查规则。"
            )
        return tools
