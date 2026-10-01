"""Shared evidence retrieval and result serialization for review tools."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def search_tagged_documents(
    knowledge_base: Any,
    docs_dir: Union[str, Path],
    query: str,
    tag: str,
    source_type: str,
) -> List[Dict[str, Any]]:
    if not knowledge_base or not any(
        page.get("tag") == tag for page in knowledge_base.pages
    ):
        return []
    try:
        result = json.loads(knowledge_base.search_keyword(query, filter_tag=tag))
    except (TypeError, ValueError):
        return []

    evidence = result.get("evidence", []) if isinstance(result, dict) else []
    if not isinstance(evidence, list):
        return []

    source_root = Path(docs_dir).name
    enriched = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["source_type"] = source_type
        record["source_path"] = (
            source_root + "/" + Path(record.get("doc_name", "")).name
        )
        page_start = record.get("page_start", record.get("page_num"))
        page_end = record.get("page_end", page_start)
        record["source_location"] = record["source_path"]
        if page_start is not None:
            record["source_location"] += f" · 第 {page_start} 页"
            if page_end is not None and page_end != page_start:
                record["source_location"] += f"–{page_end} 页"
        enriched.append(record)
    return enriched


def serialize_evidence(evidence: List[Dict[str, Any]], empty_message: str) -> str:
    return json.dumps({
        "evidence": evidence,
        "message": "" if evidence else empty_message,
    }, ensure_ascii=False)