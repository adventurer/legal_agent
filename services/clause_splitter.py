"""合同文本条款切分。"""

import re
from typing import List

from services.document_models import ContractClause


class ClauseSplitter:
    """按一级条款切分合同，保留子条款完整性。"""

    major_clause_pattern = re.compile(
        r"^(?:第[一二三四五六七八九十百千万\d]+条|\b[一二三四五六七八九十百]+[、. ])\s*(.*)$"
    )
    sign_page_pattern = re.compile(
        r"^(?:（以下无正文|以下无正文|双方签署|协议签署盖章页|甲方（盖章）|乙方（盖章）|"
        r"甲方\(盖章\)|乙方\(盖章\))"
    )

    @classmethod
    def split(cls, full_text: str) -> List[ContractClause]:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        clauses: List[ContractClause] = []
        state = "PREAMBLE"
        preamble_lines = []
        current_title = ""
        current_lines = []
        sign_page_lines = []
        counter = 0

        def append_article() -> None:
            nonlocal counter, current_lines
            if not current_title:
                return
            counter += 1
            body_content = "\n".join(current_lines).strip()
            clauses.append(
                ContractClause(
                    index=counter,
                    title=current_title,
                    content=body_content,
                    raw_text=f"{current_title}\n{body_content}",
                    clause_type="article",
                )
            )
            current_lines = []

        for line in lines:
            if cls.sign_page_pattern.search(line):
                if state == "BODY":
                    append_article()
                state = "SIGN_PAGE"
                sign_page_lines.append(line)
                continue
            if state == "SIGN_PAGE":
                sign_page_lines.append(line)
                continue

            if cls.major_clause_pattern.match(line):
                if state == "PREAMBLE":
                    if preamble_lines:
                        counter += 1
                        content = "\n".join(preamble_lines).strip()
                        clauses.append(
                            ContractClause(
                                index=counter,
                                title="合同前言与主体信息",
                                content=content,
                                raw_text=content,
                                clause_type="preamble",
                            )
                        )
                    state = "BODY"
                else:
                    append_article()
                current_title = line
            elif state == "PREAMBLE":
                preamble_lines.append(line)
            elif state == "BODY":
                current_lines.append(line)

        if state == "BODY":
            append_article()
        if sign_page_lines:
            counter += 1
            content = "\n".join(sign_page_lines).strip()
            clauses.append(
                ContractClause(
                    index=counter,
                    title="协议签署与盖章页",
                    content=content,
                    raw_text=content,
                    clause_type="sign_page",
                )
            )
        if not clauses and full_text.strip():
            content = full_text.strip()
            clauses.append(
                ContractClause(
                    index=1,
                    title="合同正文",
                    content=content,
                    raw_text=content,
                    clause_type="article",
                )
            )
        return clauses
