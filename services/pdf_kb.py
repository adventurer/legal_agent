#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: services/pdf_kb.py
职责:
1. 加载并解析 reference_docs 目录下的权威 PDF 依据（民法典、合规手册等）
2. 支持 TXT 文本缓存加速：优先从同名 .txt 缓存直接加载，无缓存或过期时解析 PDF 并生成缓存
3. 构建以 (文件名, 页码, 文本) 为基本单元的轻量级检索索引
4. 提供按类别标签 (filter_tag) 与关键词的语义/字面检索能力
"""

import os
import re
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
    """权威参考文档 PDF 本地知识库（支持 TXT 缓存加速）"""

    PAGE_SEP_PREFIX = "--- PAGE:"
    PAGE_SEP_PATTERN = re.compile(r"^--- PAGE:\s*(\d+)\s*---$")

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

    def _load_from_cache(self, cache_path: Path, doc_name: str, tag: str) -> List[Dict[str, Any]]:
        """从 TXT 缓存文件中按页还原页面数据"""
        pages = []
        current_page_num = 1
        current_text_lines: List[str] = []

        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                match = self.PAGE_SEP_PATTERN.match(stripped_line)
                if match:
                    # 归档上一页内容
                    if current_text_lines:
                        page_text = " ".join("".join(current_text_lines).split())
                        if page_text:
                            pages.append({
                                "doc_name": doc_name,
                                "page_num": current_page_num,
                                "text": page_text,
                                "tag": tag,
                            })
                        current_text_lines = []
                    current_page_num = int(match.group(1))
                else:
                    current_text_lines.append(line)

            # 归档最后一页
            if current_text_lines:
                page_text = " ".join("".join(current_text_lines).split())
                if page_text:
                    pages.append({
                        "doc_name": doc_name,
                        "page_num": current_page_num,
                        "text": page_text,
                        "tag": tag,
                    })

        return pages

    def _save_to_cache(self, cache_path: Path, pages: List[Dict[str, Any]]):
        """将解析出的页面写入 TXT 缓存，附带页码定位分隔符"""
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                for p in pages:
                    f.write(f"{self.PAGE_SEP_PREFIX} {p['page_num']} ---\n")
                    f.write(p["text"] + "\n")
        except Exception as e:
            print(f"[警告] 写入缓存文件失败: {cache_path.name}, 详情: {e}")

    def _load_documents(self):
        """扫描并加载 reference_docs 目录下的全部参考文件（优先命中缓存）"""
        if not self.docs_dir.exists():
            print(f"[警告] 知识库目录不存在: {self.docs_dir}")
            return

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        print(f"[*] 正在从 {self.docs_dir} 加载参考文档，发现 {len(pdf_files)} 个 PDF 文件...")

        for pdf_path in pdf_files:
            doc_name = pdf_path.name
            tag = self._determine_tag(doc_name)
            cache_path = pdf_path.with_suffix(".pdf.txt")

            # 1. 检查是否存在有效且未过期的缓存（缓存 mtime >= PDF mtime）
            if cache_path.exists() and cache_path.stat().st_mtime >= pdf_path.stat().st_mtime:
                try:
                    cached_pages = self._load_from_cache(cache_path, doc_name, tag)
                    if cached_pages:
                        self.pages.extend(cached_pages)
                        print(f"  [⚡缓存命中] {doc_name} -> 直接从 {cache_path.name} 读取 ({len(cached_pages)} 页)")
                        continue
                except Exception as e:
                    print(f"  [!] 读取缓存异常，重新解析 PDF: {cache_path.name}, 详情: {e}")

            # 2. 无可用缓存或缓存已失效，解析原始 PDF
            if pypdf is None:
                print(f"[警告] 未检测到 pypdf 库，无法解析 {doc_name}，请执行 `pip install pypdf`。")
                continue

            doc_pages = []
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    cleaned_text = " ".join(text.split())
                    if cleaned_text:
                        doc_pages.append({
                            "doc_name": doc_name,
                            "page_num": page_idx + 1,
                            "text": cleaned_text,
                            "tag": tag,
                        })

                if doc_pages:
                    self.pages.extend(doc_pages)
                    # 3. 异步写入/落盘缓存文件供下次直接使用
                    self._save_to_cache(cache_path, doc_pages)
                    print(f"  [+] {doc_name} 解析完成 ({len(doc_pages)} 页)，已生成缓存: {cache_path.name}")
                else:
                    print(f"  [!] {doc_name} 未提取到有效文本（可能是纯扫描图片）。")

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