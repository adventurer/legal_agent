"""网关 HTTP 客户端。"""

from typing import Any, Dict, Optional

import httpx


def check_gateway_health(base_url: str) -> Dict[str, Any]:
    """读取网关健康状态。"""
    response = httpx.get(f"{base_url}/health", timeout=2.0)
    response.raise_for_status()
    return response.json()


def upload_contract_file(
    base_url: str,
    uploaded_file: Any,
    session_id: Optional[str],
) -> Dict[str, Any]:
    """上传合同文件并返回网关响应。"""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }
    data = {"session_id": session_id} if session_id else {}
    response = httpx.post(
        f"{base_url}/api/v1/contract/upload",
        files=files,
        data=data,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()
