"""合同上传校验与临时文件管理。"""

import secrets
from pathlib import Path

from fastapi import HTTPException, UploadFile


ALLOWED_CONTRACT_EXTENSIONS = {
    ".docx",
    ".doc",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".txt",
    ".md",
}


def get_safe_extension(filename: str) -> str:
    """校验文件名并返回安全的小写扩展名。"""
    extension = Path(Path(filename or "").name).suffix.lower()
    if extension not in ALLOWED_CONTRACT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {extension or 'unknown'}")
    return extension


def create_temp_upload_path(upload_dir: Path, session_id: str, extension: str) -> Path:
    """创建不包含用户原始文件名的临时路径。"""
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f"{session_id}_{secrets.token_hex(8)}{extension}"


async def save_upload_file(file: UploadFile, destination: Path, max_bytes: int) -> None:
    """以分块方式保存上传文件并执行大小限制。"""
    bytes_read = 0
    with open(destination, "wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件大小超过限制（最大 {max_bytes} 字节）",
                )
            output.write(chunk)
