"""Document loading and TXT cache handling for the PDF knowledge base."""

import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        pypdf = None

PAGE_SEP_PREFIX = "--- PAGE:"
PAGE_SEP_PATTERN = re.compile(r"^--- PAGE:\s*(\d+)\s*---$")


def determine_tag(filename: str) -> str:
    if "法" in filename:
        return "法"
    if any(word in filename for word in ("合规", "政策", "手册")):
        return "合规"
    return "通用"


def load_cache(cache_path: Path, doc_name: str, tag: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    page_num = 1
    lines: List[str] = []
    with open(cache_path, "r", encoding="utf-8") as cache:
        for line in cache:
            match = PAGE_SEP_PATTERN.match(line.strip())
            if match:
                _append_page(pages, doc_name, page_num, lines, tag)
                lines = []
                page_num = int(match.group(1))
            else:
                lines.append(line)
    _append_page(pages, doc_name, page_num, lines, tag)
    return pages


def _append_page(
    pages: List[Dict[str, Any]], doc_name: str, page_num: int,
    lines: List[str], tag: str,
) -> None:
    text = " ".join("".join(lines).split())
    if text:
        pages.append({"doc_name": doc_name, "page_num": page_num, "text": text, "tag": tag})


def save_cache(cache_path: Path, pages: List[Dict[str, Any]]) -> None:
    try:
        with open(cache_path, "w", encoding="utf-8") as cache:
            for page in pages:
                cache.write(f"{PAGE_SEP_PREFIX} {page['page_num']} ---\n")
                cache.write(page["text"] + "\n")
    except Exception as exc:
        print(f"[警告] 写入缓存文件失败: {cache_path.name}, 详情: {exc}")


def load_pdf(pdf_path: Path, doc_name: str, tag: str) -> List[Dict[str, Any]]:
    if pypdf is None:
        print(f"[警告] 未检测到 pypdf 库，无法解析 {doc_name}，请执行 `pip install pypdf`。")
        return []
    pages: List[Dict[str, Any]] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for index, page in enumerate(reader.pages):
            text = " ".join((page.extract_text() or "").split())
            if text:
                pages.append({"doc_name": doc_name, "page_num": index + 1, "text": text, "tag": tag})
    except Exception as exc:
        print(f"[错误] 解析 PDF 失败: {doc_name}, 详情: {exc}")
    return pages


def load_documents(docs_dir: Path) -> List[Dict[str, Any]]:
    if not docs_dir.exists():
        print(f"[警告] 知识库目录不存在: {docs_dir}")
        return []
    pdf_files = list(docs_dir.glob("*.pdf"))
    print(f"[*] 正在从 {docs_dir} 加载参考文档，发现 {len(pdf_files)} 个 PDF 文件...")
    pages: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        doc_name, tag = pdf_path.name, determine_tag(pdf_path.name)
        cache_path = pdf_path.with_suffix(".pdf.txt")
        loaded = False
        if cache_path.exists() and cache_path.stat().st_mtime >= pdf_path.stat().st_mtime:
            try:
                cached = load_cache(cache_path, doc_name, tag)
                if cached:
                    pages.extend(cached)
                    loaded = True
                    print(f"  [⚡缓存命中] {doc_name} -> 直接从 {cache_path.name} 读取 ({len(cached)} 页)")
            except Exception as exc:
                print(f"  [!] 读取缓存异常，重新解析 PDF: {cache_path.name}, 详情: {exc}")
        if loaded:
            continue
        loaded_pages = load_pdf(pdf_path, doc_name, tag)
        if loaded_pages:
            pages.extend(loaded_pages)
            save_cache(cache_path, loaded_pages)
            print(f"  [+] {doc_name} 解析完成 ({len(loaded_pages)} 页)，已生成缓存: {cache_path.name}")
        else:
            print(f"  [!] {doc_name} 未提取到有效文本（可能是纯扫描图片）。")
    print(f"[+] 知识库索引构建完成，共载入 {len(pages)} 页参考内容。")
    return pages
