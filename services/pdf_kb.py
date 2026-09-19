"""Public facade for the local PDF knowledge base."""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from services.pdf_kb_loader import (
        PAGE_SEP_PATTERN,
        PAGE_SEP_PREFIX,
        determine_tag,
        load_cache,
        load_documents,
        save_cache,
    )
    from services.pdf_kb_search import search_pages
except ImportError:  # Support ``python services/pdf_kb.py`` as before.
    from pdf_kb_loader import (
        PAGE_SEP_PATTERN,
        PAGE_SEP_PREFIX,
        determine_tag,
        load_cache,
        load_documents,
        save_cache,
    )
    from pdf_kb_search import search_pages


class PDFKnowledgeBase:
    """Compatibility facade retaining the historical PDFKnowledgeBase API."""

    PAGE_SEP_PREFIX = PAGE_SEP_PREFIX
    PAGE_SEP_PATTERN = PAGE_SEP_PATTERN

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir).resolve()
        self.pages: List[Dict[str, Any]] = load_documents(self.docs_dir)

    def _determine_tag(self, filename: str) -> str:
        return determine_tag(filename)

    def _load_from_cache(self, cache_path: Path, doc_name: str, tag: str):
        return load_cache(cache_path, doc_name, tag)

    def _save_to_cache(self, cache_path: Path, pages: List[Dict[str, Any]]) -> None:
        save_cache(cache_path, pages)

    def _load_documents(self) -> None:
        self.pages = load_documents(self.docs_dir)

    def search_keyword(
        self, query: str, filter_tag: Optional[str] = None,
        top_k: int = 2, max_snippet_len: int = 350,
    ) -> str:
        return search_pages(self.pages, query, filter_tag, top_k, max_snippet_len)
