#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: services/doc_loader.py
职责:
1. 多源合同解析：支持 Word (.docx, .doc)、PDF、图片与纯文本
2. 生产级双轨 PDF 提取：优先提取矢量文字；扫描件自动进入 OCR 兜底
3. 将解析后的文本交给独立条款切分器处理
"""

import os
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any

from services.clause_splitter import ClauseSplitter
from services.document_models import ContractClause

# 1. 基础文档处理依赖
try:
    from docx import Document
except ImportError:
    Document = None

try:
    import fitz  # PyMuPDF（PDF 提取与光栅化核心）
except ImportError:
    fitz = None

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        pypdf = None

# 2. 图片与 OCR 依赖
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from rapidocr_onnxruntime import RapidOCR
    _rapid_ocr_engine = RapidOCR()
except ImportError:
    _rapid_ocr_engine = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class DocumentLoader:
    """合同文档多模态解析与切分器 (工业增强版)"""

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    SUPPORTED_WORD_EXTS = {".docx", ".doc"}
    SUPPORTED_TEXT_EXTS = {".txt", ".md"}

    def __init__(self):
        pass

    def extract_text_from_docx(self, file_path: Path) -> str:
        """从 Word 文档中提取段落与表格文本"""
        if Document is None:
            raise ImportError("未安装 python-docx，请执行: pip install python-docx")

        doc = Document(str(file_path))
        lines = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                lines.append(text)

        for table in doc.tables:
            for row in table.rows:
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_texts:
                    unique_texts = list(dict.fromkeys(row_texts))
                    lines.append(" | ".join(unique_texts))

        return "\n".join(lines)

    def _convert_doc_to_docx(self, file_path: Path, output_dir: Path) -> Path:
        """将旧版 .doc 转为临时 .docx，Linux/WSL 优先使用 LibreOffice。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        converted_path = output_dir / f"{file_path.stem}.docx"
        conversion_errors = []

        # Linux/WSL 的标准方案。为每次转换创建独立 profile，避免并发转换互相锁定。
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            profile_dir = output_dir / "lo-profile"
            try:
                result = subprocess.run(
                    [
                        soffice,
                        f"-env:UserInstallation={profile_dir.as_uri()}",
                        "--headless",
                        "--convert-to",
                        "docx:Office Open XML Text",
                        "--outdir",
                        str(output_dir),
                        str(file_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0 and converted_path.exists():
                    return converted_path
                conversion_errors.append(
                    f"LibreOffice 返回码 {result.returncode}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                conversion_errors.append(f"LibreOffice: {exc}")

        # Windows 原生兼容方案；不参与 Linux/WSL 的默认路径。
        if os.name == "nt":
            word = None
            document = None
            try:
                import win32com.client

                word = win32com.client.DispatchEx("Word.Application")
                word.Visible = False
                word.DisplayAlerts = False
                document = word.Documents.Open(
                    str(file_path),
                    ReadOnly=True,
                    AddToRecentFiles=False,
                    ConfirmConversions=False,
                )
                document.SaveAs2(str(converted_path), FileFormat=16)
                if converted_path.exists():
                    return converted_path
            except (ImportError, OSError, RuntimeError) as exc:
                conversion_errors.append(f"Word COM: {exc}")
            finally:
                if document is not None:
                    try:
                        document.Close(False)
                    except Exception:
                        pass
                if word is not None:
                    try:
                        word.Quit()
                    except Exception:
                        pass

        details = "；".join(error for error in conversion_errors if error)
        raise RuntimeError(
            "无法读取旧版 .doc 文件。Linux/WSL 请安装 LibreOffice（soffice），"
            "Windows 可安装 Microsoft Word 并安装 pywin32。"
            + (f" 详情: {details}" if details else "")
        )

    def extract_text_from_doc(self, file_path: Path) -> str:
        """转换旧版 Word 文档后复用 DOCX 解析逻辑。"""
        with tempfile.TemporaryDirectory(prefix="legal_agent_doc_") as temp_dir:
            docx_path = self._convert_doc_to_docx(file_path, Path(temp_dir))
            return self.extract_text_from_docx(docx_path)

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """
        高可用双轨 PDF 提取方案：
        1. 优先使用 PyMuPDF (fitz) 提取矢量文字。
        2. 扫描件/图片页判定：若单页有效文字少于 15 字符，光栅化为 200 DPI 图片，送入 OCR 兜底。
        3. 次选 pypdf 降级运行。
        """
        lines = []
        path_str = str(file_path)

        # 方案 A: PyMuPDF (fitz)
        if fitz is not None:
            try:
                doc = fitz.open(path_str)
                for page_idx, page in enumerate(doc):
                    page_text = page.get_text("text").strip()
                    # 统计中文字符与英文字符数，过滤掉仅有空白页或页码干扰
                    valid_chars = re.findall(r'[\u4e00-\u9fa5\w]', page_text)
                    
                    if len(valid_chars) >= 15:
                        for line in page_text.splitlines():
                            if line.strip():
                                lines.append(line.strip())
                    else:
                        # 触发 OCR 兜底：纯图片扫描页或复杂印章页
                        print(f"[*] PDF 第 {page_idx+1} 页文字稀少（疑似扫描件），启动 OCR 光栅化识别...", flush=True)
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = self._do_ocr(img)
                        for line in ocr_text.splitlines():
                            if line.strip():
                                lines.append(line.strip())
                if lines:
                    return "\n".join(lines)
            except Exception as e:
                print(f"[警告] PyMuPDF 解析失败: {e}，尝试降级引擎...", flush=True)

        # 方案 B: 基础 pypdf 降级
        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(path_str)
                for page in reader.pages:
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        if line.strip():
                            lines.append(line.strip())
                if lines:
                    return "\n".join(lines)
            except Exception as e:
                print(f"[警告] pypdf 解析失败: {e}", flush=True)

        if not lines:
            raise RuntimeError(
                f"PDF 文件解析失败或为空: {file_path.name}。\n"
                f"请确保安装了 PyMuPDF 与 RapidOCR:\n"
                f"pip install pymupdf rapidocr-onnxruntime pillow"
            )

        return "\n".join(lines)

    def _do_ocr(self, img_obj: Any) -> str:
        """统一底层 OCR 执行器，内置防报错机制"""
        if _rapid_ocr_engine is not None:
            try:
                # 兼容无需显式安装 numpy 的安全转换
                if hasattr(img_obj, "convert"):
                    img_rgb = img_obj.convert("RGB")
                    with io.BytesIO() as buf:
                        img_rgb.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    result, _ = _rapid_ocr_engine(img_bytes)
                else:
                    result, _ = _rapid_ocr_engine(img_obj)

                if result:
                    return "\n".join([item[1].strip() for item in result if item[1].strip()])
                return ""
            except Exception as e:
                print(f"[警告] RapidOCR 执行异常: {e}，尝试备选 OCR 引擎...", flush=True)

        if pytesseract is not None:
            try:
                return pytesseract.image_to_string(img_obj, lang="chi_sim+eng")
            except Exception as e:
                print(f"[警告] pytesseract 识别失败: {e}", flush=True)

        return ""

    def extract_text_from_image(self, file_path: Path) -> str:
        print(file_path)
        """从原生图片文件中提取文字"""
        if Image is None:
            raise ImportError("未安装 Pillow 库，请执行: pip install pillow")
        img = Image.open(file_path)
        text = self._do_ocr(img)
        if not text.strip():
            raise RuntimeError(f"未能从图片中提取到有效文本: {file_path.name}")
        return text

    def extract_text(self, file_path: str | Path) -> str:
        """通用格式路由分发"""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()

        if suffix == ".docx":
            raw_text = self.extract_text_from_docx(path)
        elif suffix == ".doc":
            raw_text = self.extract_text_from_doc(path)
        elif suffix == ".pdf":
            raw_text = self.extract_text_from_pdf(path)
        elif suffix in self.SUPPORTED_IMAGE_EXTS:
            print(f"[*] 检测到图片格式合同 ({suffix})，正在启动 OCR 文本识别...", flush=True)
            raw_text = self.extract_text_from_image(path)
        elif suffix in self.SUPPORTED_TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        else:
            supported = sorted(self.SUPPORTED_WORD_EXTS | {".pdf"} | self.SUPPORTED_TEXT_EXTS | self.SUPPORTED_IMAGE_EXTS)
            raise ValueError(f"不支持的文件格式: {suffix}，当前支持格式: {supported}")

        return self._normalize_text(raw_text)

    def _normalize_text(self, text: str) -> str:
        """清洗排版乱码与冗余字符"""
        if not text:
            return ""
        text = text.replace("\t", " ").replace("\u3000", " ")
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    def split_into_clauses(self, full_text: str) -> List[ContractClause]:
        """
        层次化状态机切片：
        1. 严格以“第一条/一、”为大条款边界；
        2. 1.1、1.2、(1)、(2) 等子项全部合并保留在所属大条款正文中，防语义碎裂；
        3. 完整识别并隔离前言（PREAMBLE）与签署盖章页（SIGN_PAGE）。
        """
        return ClauseSplitter.split(full_text)

    def load_and_split(self, file_path: str | Path) -> List[ContractClause]:
        """对外主接口"""
        text = self.extract_text(file_path)
        return self.split_into_clauses(text)
