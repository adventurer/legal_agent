"""合同修订 API 客户端。"""

from typing import Any, Dict, List

import httpx


def rewrite_contract(
    base_url: str,
    clauses: List[Dict[str, Any]],
    review_report: str,
    selected_indices: List[int],
) -> Dict[str, Any]:
    response = httpx.post(
        f"{base_url}/api/v1/contract/rewrite",
        json={
            "clauses": clauses,
            "review_report": review_report,
            "selected_indices": selected_indices,
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()
