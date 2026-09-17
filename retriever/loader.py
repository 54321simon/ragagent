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

def load_docx(path: str) -> list[dict]:
    """解析 Word 文档。返回 [{doc_id, doc_name, page, text}]。
    docx 没有页码概念，统一 page=1。"""
    import docx
    doc = docx.Document(path)
    doc_name = os.path.basename(path)
    doc_id = doc_name.rsplit(".", 1)[0]
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text:
        return []
    return [{
        "doc_id": doc_id,
        "doc_name": doc_name,
        "page": 1,
        "text": text,
    }]


def load_txt(path: str) -> list[dict]:
    """解析 TXT。page=1。"""
    doc_name = os.path.basename(path)
    doc_id = doc_name.rsplit(".", 1)[0]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    return [{"doc_id": doc_id, "doc_name": doc_name, "page": 1, "text": text}]


def load_md(path: str) -> list[dict]:
    """解析 Markdown。page=1。"""
    return load_txt(path)  # md 就是文本


def load_any(path: str) -> list[dict]:
    """根据扩展名自动选择加载器。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"不支持的文件类型: {ext}")