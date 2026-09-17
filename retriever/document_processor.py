"""文档解析：PDF、Word读取 + 文本切分"""
import pymupdf
from docx import Document
from typing import List, Dict


def load_file(file_path: str) -> str:
    """读取PDF / Word文档"""
    if file_path.lower().endswith(".pdf"):
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    elif file_path.lower().endswith(".txt") or file_path.lower().endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise NotImplementedError("仅支持pdf,docx,txt,md")


def split_text(text: str, chunk_size: int = 512, overlap: int = 80) -> List[Dict]:
    """固定大小滑动窗口切片，返回chunk列表，附带chunk_id"""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "chunk_id": f"chunk_{idx}",
            "text": chunk_text
        })
        idx += 1
        start = start + chunk_size - overlap
    return chunks