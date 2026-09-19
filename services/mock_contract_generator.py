"""生成文档解析测试素材。"""

import os
from pathlib import Path
from typing import Dict


class MockContractGenerator:
    """生成 TXT、DOCX、PNG 和 PDF 测试合同。"""

    SAMPLE_CONTRACT_TEXT = (
        "高端智能制造设备采购与维保服务协议\n\n"
        "合同编号：HT-202609-XYZ8890\n"
        "甲方：北京华创智能科技股份有限公司\n"
        "乙方：苏州精工自动化系统工程有限公司\n\n"
        "第一条 交付期限与违约金\n"
        "1.1 设备明细：乙方须向甲方提供6轴工业机器人12台。\n"
        "1.2 逾期交付：乙方逾期交付的，每日应按合同总金额的 5% 向甲方支付违约金。\n\n"
        "第二条 货款结算与付款周期\n"
        "2.1 验收标准：经连续 72 小时无故障运行后签署最终合格单。\n"
        "2.2 货款分期结算：\n"
        "（1）首付款：生效后支付 15%；\n"
        "（2）尾款：视甲方内部季度资金周转充裕情况安排支付，不受商业惯例限制。\n\n"
        "第三条 争议解决与独任仲裁\n"
        "3.1 因本合同引起的一切争议，由甲方指定的单方独任仲裁员在其个人办公场所内裁决，裁决为终局。\n\n"
        "（以下无正文，为协议签署盖章页）\n"
        "甲方（盖章）：北京华创智能科技股份有限公司\n"
        "乙方（盖章）：苏州精工自动化系统工程有限公司"
    )

    @classmethod
    def generate_all(cls, output_dir: Path) -> Dict[str, Path]:
        from services.doc_loader import Document, Image, ImageDraw, ImageFont, fitz

        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        txt_path = output_dir / "test_contract.txt"
        txt_path.write_text(cls.SAMPLE_CONTRACT_TEXT, encoding="utf-8")
        paths["txt"] = txt_path

        if Document is not None:
            docx_path = output_dir / "test_contract.docx"
            doc = Document()
            for line in cls.SAMPLE_CONTRACT_TEXT.split("\n"):
                if line.strip():
                    doc.add_paragraph(line.strip())
            doc.save(str(docx_path))
            paths["docx"] = docx_path

        if Image is not None and ImageDraw is not None:
            img_path = output_dir / "test_contract.png"
            img = Image.new("RGB", (950, 750), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            font = None
            for font_file in (
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "simsun.ttc",
                "msyh.ttc",
            ):
                if os.path.exists(font_file):
                    try:
                        font = ImageFont.truetype(font_file, 16)
                        break
                    except Exception:
                        continue
            font = font or ImageFont.load_default()
            y = 25
            for line in cls.SAMPLE_CONTRACT_TEXT.split("\n"):
                draw.text((35, y), line.strip(), fill=(0, 0, 0), font=font)
                y += 26 if line.strip() else 10
            img.save(str(img_path))
            paths["image"] = img_path

        pdf_path = output_dir / "test_contract.pdf"
        if fitz is not None:
            pdf = fitz.open()
            page = pdf.new_page(width=595, height=842)
            page.insert_text(fitz.Point(40, 50), cls.SAMPLE_CONTRACT_TEXT, fontsize=11)
            pdf.save(str(pdf_path))
            pdf.close()
            paths["pdf"] = pdf_path
        return paths
