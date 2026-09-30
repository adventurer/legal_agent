"""Document loading and TXT cache handling for the PDF knowledge base."""

import hashlib
import json
import os
import re
import tempfile
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
ARTICLE_PATTERN = re.compile(
    r"(?<![\u3400-\u9fffA-Za-z0-9])(?P<number>第\s*[零〇一二三四五六七八九十百千万两0-9]+\s*条)"
    r"(?P<heading>\s*(?:【[^】]{1,80}】)?)(?=\s*[\u3400-\u9fff])"
)
POLICY_NAME_HINTS = ("合规", "政策", "手册", "policy", "compliance")
LAW_NAME_HINTS = (
    "法", "minfadian", "minshishusongfa", "zhongcaifa", "civilcode",
    "civil_code", "arbitrationlaw", "arbitration_law", ".law",
)


def determine_tag(filename: str) -> str:
    name = Path(filename).stem.lower()
    if any(word in name for word in POLICY_NAME_HINTS):
        return "合规"
    if any(word in name for word in LAW_NAME_HINTS):
        return "法"
    return "通用"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def save_cache(cache_path: Path, pages: List[Dict[str, Any]]) -> bool:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent,
            prefix=f"{cache_path.name}.", suffix=".tmp", delete=False,
        ) as cache:
            temp_path = Path(cache.name)
            for page in pages:
                cache.write(f"{PAGE_SEP_PREFIX} {page['page_num']} ---\n")
                cache.write(page["text"] + "\n")
            cache.flush()
            os.fsync(cache.fileno())
        os.replace(temp_path, cache_path)
        return True
    except Exception as exc:
        print(f"[警告] 写入缓存文件失败: {cache_path.name}, 详情: {exc}")
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False


def split_pages_into_articles(pages: List[Dict[str, Any]], source_hash: str) -> List[Dict[str, Any]]:
    """Split extracted page text into article-sized searchable evidence records."""
    articles: List[Dict[str, Any]] = []
    current_article = None
    for page in pages:
        text = page.get("text", "")
        matches = list(ARTICLE_PATTERN.finditer(text))
        segments = []
        if matches:
            prefix = text[:matches[0].start()].strip()
            if prefix:
                segments.append((None, "", prefix))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                article_text = text[match.start():end].strip()
                heading = match.group("heading").strip()
                title_match = re.match(r"【([^】]+)】", heading)
                segments.append((re.sub(r"\s+", "", match.group("number")), title_match.group(1) if title_match else "", article_text))
        else:
            segments.append((None, "", text.strip()))

        for segment_index, (article_no, title, article_text) in enumerate(segments):
            if not article_text:
                continue
            # A page may begin with the remainder of the previous article. Join
            # that text to the same evidence record so the dialog shows the full article.
            if article_no is None and current_article is not None:
                current_article["text"] += article_text
                current_article["page_end"] = page["page_num"]
                if page["page_num"] not in current_article["source_pages"]:
                    current_article["source_pages"].append(page["page_num"])
                continue
            seed = f"{source_hash}|{page['doc_name']}|{page['page_num']}|{article_no or 'fragment'}|{article_text}"
            evidence_id = "EV" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12].upper()
            record = {
                "id": evidence_id, "doc_name": page["doc_name"], "tag": page["tag"],
                "page_num": page["page_num"], "page_start": page["page_num"],
                "page_end": page["page_num"], "article_no": article_no,
                "source_pages": [page["page_num"]], "title": title, "text": article_text,
            }
            articles.append(record)
            if article_no is not None:
                current_article = record
    for record in articles:
        seed = f"{source_hash}|{record['doc_name']}|{record['article_no'] or 'fragment'}|{record['page_start']}|{record['text']}"
        record["id"] = "EV" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12].upper()
    return articles


def load_article_cache(cache_path: Path, source_hash: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") == 1 and data.get("source_sha256") == source_hash:
            return data.get("articles", [])
    except (OSError, ValueError, TypeError):
        pass
    return []


def save_article_cache(cache_path: Path, articles: List[Dict[str, Any]], source_hash: str) -> None:
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps({"version": 1, "source_sha256": source_hash, "articles": articles}, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, cache_path)
    except OSError as exc:
        print(f"[警告] 写入法条缓存失败: {cache_path.name}, 详情: {exc}")
        temp_path.unlink(missing_ok=True)


def load_pdf(pdf_path: Path, doc_name: str, tag: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    try:
        import fitz

        from services.doc_loader import DocumentLoader

        ocr_loader = DocumentLoader()
        pdf = fitz.open(str(pdf_path))
        for index, page in enumerate(pdf):
            text = " ".join((page.get_text("text") or "").split())
            if len(re.findall(r"[\u3400-\u9fff\w]", text)) < 20:
                try:
                    from PIL import Image

                    pix = page.get_pixmap(dpi=180, colorspace=fitz.csRGB, alpha=False)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = " ".join(ocr_loader._do_ocr(image).split())
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except (ImportError, RuntimeError, ValueError) as exc:
                    print(f"[警告] {doc_name} 第 {index + 1} 页 OCR 不可用: {exc}")
            if text:
                pages.append({"doc_name": doc_name, "page_num": index + 1, "text": text, "tag": tag})
        if pages:
            return pages
    except ImportError:
        pass
    except Exception as exc:
        print(f"[警告] PyMuPDF/OCR 解析失败，尝试 pypdf: {doc_name}, 详情: {exc}")
    pages = []

    if pypdf is None:
        print(f"[警告] 未检测到 pypdf 库，无法解析 {doc_name}，请执行 `pip install pypdf`。")
        return pages
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
    law_text_files = list(docs_dir.glob("*.law.txt"))
    print(f"[*] 正在从 {docs_dir} 加载参考文档，发现 {len(pdf_files)} 个 PDF 和 {len(law_text_files)} 个法律文本文件...")
    pages: List[Dict[str, Any]] = []
    articles: List[Dict[str, Any]] = []
    for text_path in law_text_files:
        doc_name, tag = text_path.name, determine_tag(text_path.name)
        source_hash = _sha256(text_path)
        article_cache_path = text_path.with_suffix(".articles.json")
        cached_articles = load_article_cache(article_cache_path, source_hash)
        if cached_articles:
            articles.extend(cached_articles)
            print(f"  [⚡法条缓存命中] {doc_name} ({len(cached_articles)} 条/片段)")
            continue
        text = text_path.read_text(encoding="utf-8-sig")
        page = {"doc_name": doc_name, "page_num": 1, "text": " ".join(text.split()), "tag": tag}
        loaded_articles = split_pages_into_articles([page], source_hash)
        articles.extend(loaded_articles)
        save_article_cache(article_cache_path, loaded_articles, source_hash)
        print(f"  [+] {doc_name} 已载入 ({len(loaded_articles)} 条/片段)")
    for pdf_path in pdf_files:
        doc_name, tag = pdf_path.name, determine_tag(pdf_path.name)
        source_hash = _sha256(pdf_path)
        article_cache_path = pdf_path.with_suffix(".articles.json")
        cached_articles = load_article_cache(article_cache_path, source_hash)
        if cached_articles:
            articles.extend(cached_articles)
            print(f"  [⚡法条缓存命中] {doc_name} ({len(cached_articles)} 条/片段)")
            continue
        cache_path = pdf_path.with_suffix(".pdf.txt")
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
        loaded = False
        if cache_path.exists() and meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("source_sha256") == source_hash
                    and metadata.get("cache_sha256") == _sha256(cache_path)
                ):
                    cached = load_cache(cache_path, doc_name, tag)
                else:
                    cached = []
                if cached:
                    loaded_articles = split_pages_into_articles(cached, source_hash)
                    articles.extend(loaded_articles)
                    save_article_cache(article_cache_path, loaded_articles, source_hash)
                    loaded = True
                    print(f"  [⚡缓存命中] {doc_name} -> {len(loaded_articles)} 条/片段")
            except Exception as exc:
                print(f"  [!] 读取缓存异常，重新解析 PDF: {cache_path.name}, 详情: {exc}")
        if loaded:
            continue
        loaded_pages = load_pdf(pdf_path, doc_name, tag)
        if loaded_pages:
            loaded_articles = split_pages_into_articles(loaded_pages, source_hash)
            articles.extend(loaded_articles)
            save_article_cache(article_cache_path, loaded_articles, source_hash)
            if save_cache(cache_path, loaded_pages):
                try:
                    metadata = json.dumps({
                        "source_sha256": _sha256(pdf_path),
                        "cache_sha256": _sha256(cache_path),
                    })
                    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
                    meta_tmp.write_text(metadata, encoding="utf-8")
                    os.replace(meta_tmp, meta_path)
                except OSError as exc:
                    print(f"[警告] 知识库缓存元数据写入失败: {meta_path.name}, 详情: {exc}")
            print(f"  [+] {doc_name} 解析完成 ({len(loaded_pages)} 页)，已生成缓存: {cache_path.name}")
        else:
            print(f"  [!] {doc_name} 未提取到有效文本（可能是纯扫描图片）。")
    print(f"[+] 知识库索引构建完成，共载入 {len(articles)} 条/片段参考内容。")
    return articles
