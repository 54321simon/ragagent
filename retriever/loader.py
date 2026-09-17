"""PDF 加载器。用 pymupdf（不是 fitz）。"""
import pymupdf
import os


def load_pdf(path: str) -> list[dict]:
    """返回 [{doc_id, doc_name, page, text}]，page 从 1 开始。"""
    doc = pymupdf.open(path)
    doc_name = os.path.basename(path)
    doc_id = doc_name.replace(".pdf", "")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": i + 1,
                "text": text,
            })
    doc.close()
    return pages