#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: services/pdf_kb.py
职责:
1. 加载并解析 reference_docs 目录下的权威 PDF 依据（民法典、合规手册等）
2. 构建以 (文件名, 页码, 文本) 为基本单元的轻量级检索索引
3. 提供按类别标签 (filter_tag) 与关键词的语义/字面检索能力
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        pypdf = None


class PDFKnowledgeBase:
    """权威参考文档 PDF 本地知识库"""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir).resolve()
        # 页面文档结构: [{"doc_name": str, "page_num": int, "text": str, "tag": str}]
        self.pages: List[Dict[str, Any]] = []
        self._load_documents()

    def _determine_tag(self, filename: str) -> str:
        """根据文件名推断文档属性标签"""
        if "法" in filename:
            return "法"
        if "合规" in filename or "政策" in filename or "手册" in filename:
            return "合规"
        return "通用"

    def _load_documents(self):
        """扫描并加载 reference_docs 目录下的全部 PDF 文件"""
        if not self.docs_dir.exists():
            print(f"[警告] 知识库目录不存在: {self.docs_dir}")
            return

        if pypdf is None:
            print("[警告] 未检测到 pypdf 或 PyPDF2 库，请执行 `pip install pypdf` 启用 PDF 解析。")
            return

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        print(f"[*] 正在从 {self.docs_dir} 加载参考文档，发现 {len(pdf_files)} 个 PDF 文件...")

        for pdf_path in pdf_files:
            doc_name = pdf_path.name
            tag = self._determine_tag(doc_name)
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    # 清洗空白字符并过滤空页
                    cleaned_text = " ".join(text.split())
                    if cleaned_text:
                        self.pages.append({
                            "doc_name": doc_name,
                            "page_num": page_idx + 1,
                            "text": cleaned_text,
                            "tag": tag,
                        })
            except Exception as e:
                print(f"[错误] 解析 PDF 失败: {doc_name}, 详情: {e}")

        print(f"[+] 知识库索引构建完成，共载入 {len(self.pages)} 页参考内容。")

    def search_keyword(
        self,
        query: str,
        filter_tag: Optional[str] = None,
        top_k: int = 2,
        max_snippet_len: int = 350,
    ) -> str:
        """
        根据关键词和标签检索法规或合规内容
        :param query: 检索关键词（如 '违约金', '争议解决', '付款'）
        :param filter_tag: 标签过滤（'法' 或 '合规'）
        :param top_k: 返回最相关的段落数量
        :param max_snippet_len: 单个段落最大截断长度
        :return: 格式化的依据文本，附带文件名与页码
        """
        if not self.pages:
            return f"未找到任何与【{query}】相关的参考依据（知识库为空或无有效页面）。"

        # 简单高效的词频与包含度打分算法
        candidates = []
        keywords = [kw.strip() for kw in query.split() if kw.strip()]
        if not keywords:
            keywords = [query.strip()]

        for item in self.pages:
            # 标签过滤
            if filter_tag and filter_tag not in item["tag"] and filter_tag not in item["doc_name"]:
                continue

            # 评分：词频加权
            score = sum(item["text"].count(kw) for kw in keywords)
            if score > 0:
                candidates.append((score, item))

        if not candidates:
            tag_desc = f"[{filter_tag}类]" if filter_tag else ""
            return f"在权威参考文档 {tag_desc} 中未检索到与【{query}】相关的法条或合规规定。"

        # 按命中得分降序排序，取前 top_k 项
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:top_k]

        results = []
        for idx, (_, item) in enumerate(selected, 1):
            text = item["text"]
            # 找到首个命中关键词的位置，以该位置为中心截取片段
            first_kw_pos = -1
            for kw in keywords:
                pos = text.find(kw)
                if pos != -1:
                    first_kw_pos = pos
                    break

            if first_kw_pos != -1:
                start = max(0, first_kw_pos - 50)
                end = min(len(text), start + max_snippet_len)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
            else:
                snippet = text[:max_snippet_len] + "..."

            results.append(
                f"依据 [{idx}] 《{item['doc_name']}》 第 {item['page_num']} 页:\n{snippet}"
            )

        return "\n\n".join(results)