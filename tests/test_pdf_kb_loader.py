from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import pdf_kb_loader


def test_startup_only_loads_canonical_law_txt(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "minfadian.pdf"
    pdf_path.write_bytes(b"pdf fixture")
    (tmp_path / "minfadian.txt").write_text("不得作为法规来源加载。", encoding="utf-8")
    law_text = tmp_path / "minfadian.law.txt"
    law_text.write_text("第一条 规范法规 TXT 是法律知识来源。", encoding="utf-8")

    def law_pdf_must_not_be_read(*_args):
        raise AssertionError("law PDFs must not be read during knowledge startup")

    monkeypatch.setattr(pdf_kb_loader, "load_pdf", law_pdf_must_not_be_read)
    loaded = pdf_kb_loader.load_documents(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["doc_name"] == law_text.name
    assert "法律知识来源" in loaded[0]["text"]


def test_startup_ignores_temporary_upload_files(tmp_path: Path, monkeypatch):
    (tmp_path / ".kb-upload-123.pdf").write_bytes(b"temporary pdf")
    (tmp_path / ".kb-upload-123.txt").write_text("temporary text", encoding="utf-8")
    (tmp_path / "valid.law.txt").write_text("第一条 正式法规。", encoding="utf-8")

    def temporary_pdf_must_not_be_read(*_args):
        raise AssertionError("temporary upload PDF must not be indexed")

    monkeypatch.setattr(pdf_kb_loader, "load_pdf", temporary_pdf_must_not_be_read)
    loaded = pdf_kb_loader.load_documents(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["doc_name"] == "valid.law.txt"


def test_markdown_sources_preserve_formatting_and_rebuild_flattened_cache(tmp_path: Path):
    source = tmp_path / "合规_审查说明.md"
    markdown_text = "# 审查重点\n\n- 核对交付日期\n- **确认付款条件**"
    source.write_text(markdown_text, encoding="utf-8")

    old_hash = pdf_kb_loader._sha256(source)
    flattened_text = " ".join(markdown_text.split())
    old_articles = pdf_kb_loader.split_pages_into_articles([{
        "doc_name": source.name,
        "page_num": 1,
        "text": flattened_text,
        "tag": "合规",
    }], old_hash)
    cache_path = source.with_suffix(".articles.json")
    assert pdf_kb_loader.save_article_cache(cache_path, old_articles, old_hash)

    loaded = pdf_kb_loader.load_documents(tmp_path)

    assert loaded[0]["text"] == markdown_text
    current_hash = pdf_kb_loader.hashlib.sha256(
        f"{old_hash}:markdown-preserved-lines-v1".encode("utf-8")
    ).hexdigest()
    rebuilt_articles = pdf_kb_loader.load_article_cache(cache_path, current_hash)
    assert rebuilt_articles[0]["text"] == markdown_text


def test_pdf_cache_path_uses_law_suffix_only_for_law_documents(tmp_path: Path):
    assert pdf_kb_loader.pdf_text_cache_path(tmp_path / "minfadian.pdf").name == "minfadian.pdf.law.txt"
    assert pdf_kb_loader.pdf_text_cache_path(tmp_path / "合规_指南.pdf").name == "合规_指南.pdf.txt"


def test_uploaded_law_sources_are_converted_to_paginated_law_txt(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "uploaded-law.pdf"
    pdf_path.write_bytes(b"pdf fixture")
    pdf_txt = tmp_path / "uploaded-law.law.txt"
    pages = [{
        "doc_name": pdf_path.name,
        "page_num": 1,
        "text": "第一条 PDF 转换为 law.txt。",
        "tag": "法",
    }]
    monkeypatch.setattr(pdf_kb_loader, "load_pdf", lambda *_: pages)

    assert pdf_kb_loader.convert_law_source_to_txt(pdf_path, pdf_txt, pdf_path.name)
    assert pdf_kb_loader.load_cache(pdf_txt, pdf_txt.name, "法")[0]["text"].endswith("law.txt。")
    assert (tmp_path / "uploaded-law.law.articles.json").is_file()

    text_path = tmp_path / "uploaded-law.md"
    text_path.write_text("第二条 Markdown 转换为 law.txt。", encoding="utf-8")
    text_txt = tmp_path / "uploaded-markdown.law.txt"
    assert pdf_kb_loader.convert_law_source_to_txt(text_path, text_txt, text_path.name)
    assert "Markdown 转换" in pdf_kb_loader.load_cache(text_txt, text_txt.name, "法")[0]["text"]


def test_law_txt_is_authoritative_when_article_cache_exists(tmp_path: Path, monkeypatch):
    law_text = tmp_path / "sample.law.txt"
    law_text.write_text("第一条 法规 TXT 是法律知识来源。", encoding="utf-8")
    pdf_kb_loader.load_documents(tmp_path)
    article_cache = tmp_path / "sample.law.articles.json"
    assert article_cache.is_file()

    original_split = pdf_kb_loader.split_pages_into_articles

    def article_cache_should_be_reused(*_args):
        raise AssertionError("valid article cache should be reused")

    monkeypatch.setattr(pdf_kb_loader, "split_pages_into_articles", article_cache_should_be_reused)
    loaded = pdf_kb_loader.load_documents(tmp_path)
    assert "法律知识来源" in loaded[0]["text"]

    monkeypatch.setattr(pdf_kb_loader, "split_pages_into_articles", original_split)
    law_text.write_text("第二条 更新后的法规 TXT。", encoding="utf-8")
    updated = pdf_kb_loader.load_documents(tmp_path)
    assert "更新后的法规" in updated[0]["text"]


def test_law_txt_related_files_include_pdf_and_all_cache_generations(tmp_path: Path):
    law_text = tmp_path / "法典_minfadian.law.txt"
    related = set(pdf_kb_loader.law_text_related_files(law_text))

    assert law_text.with_suffix(".articles.json") in related
    assert tmp_path / "法典_minfadian.pdf" in related
    assert tmp_path / "法典_minfadian.articles.json" in related
    assert tmp_path / "法典_minfadian.pdf.law.txt" in related
    assert tmp_path / "法典_minfadian.pdf.law.txt.meta.json" in related
    assert tmp_path / "法典_minfadian.pdf.txt" in related
    assert tmp_path / "法典_minfadian.pdf.txt.meta.json" in related