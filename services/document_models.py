"""文档领域模型。"""

from typing import Any, Dict


class ContractClause:
    """单个合同条款结构体。"""

    def __init__(
        self,
        index: int,
        title: str,
        content: str,
        raw_text: str,
        clause_type: str = "article",
    ):
        self.index = index
        self.title = title
        self.content = content
        self.raw_text = raw_text
        self.clause_type = clause_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "content": self.content,
            "raw_text": self.raw_text,
            "type": self.clause_type,
        }

    def __repr__(self) -> str:
        return f"<Clause {self.index} [{self.clause_type}]: {self.title[:18]}... ({len(self.content)} 字)>"
