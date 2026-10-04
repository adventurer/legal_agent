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
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            error_body = exc.response.json()
        except (ValueError, AttributeError):
            error_body = None
        if isinstance(error_body, dict) and error_body.get("detail"):
            raise httpx.HTTPStatusError(
                f"{exc} - {error_body['detail']}",
                request=exc.request,
                response=exc.response,
            ) from exc
        raise
    return response.json()
