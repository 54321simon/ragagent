"""PDF 加载器。用 pymupdf。自动过滤页眉/页脚/版权声明。"""
import pymupdf
import os
from collections import Counter


COPYRIGHT_PATTERNS = [
    "provided proper attribution",
    "hereby grants permission",
    "reproduce the tables",
    "solely for use",
    "journalistic or scholarly",
    "all rights reserved",
    "copyright",
    "arxiv:",
    "preprint",
    "under review",
    "licensed under",
    "creative commons",
]


def _is_copyright_line(line: str) -> bool:
    low = line.lower().strip()
    return any(pat in low for pat in COPYRIGHT_PATTERNS)


def load_pdf(path: str) -> list[dict]:
    """返回 [{doc_id, doc_name, page, text}]。
    自动过滤：页眉/页脚（3 页以上重复短行）+ 版权声明。"""
    doc = pymupdf.open(path)
    doc_name = os.path.basename(path)
    doc_id = doc_name.replace(".pdf", "")

    raw_pages = []
    for i, page in enumerate(doc):
        raw_pages.append({"page": i + 1, "text": page.get_text()})
    doc.close()

    # 过滤 1：页眉/页脚
    line_counter = Counter()
    for p in raw_pages:
        for l in p["text"].split("\n"):
            l = l.strip()
            if l and len(l) < 150:
                line_counter[l] += 1
    repeated = {line for line, cnt in line_counter.items() if cnt >= 3}

    # 过滤 2：版权声明 + 重复行
    pages = []
    for p in raw_pages:
        lines = p["text"].split("\n")
        filtered = []
        for l in lines:
            s = l.strip()
            if not s:
                continue
            if s in repeated:
                continue
            if _is_copyright_line(s):
                continue
            filtered.append(l)
        text = "\n".join(filtered).strip()
        if text:
            pages.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": p["page"],
                "text": text,
            })
    return pages


def load_docx(path: str) -> list[dict]:
    import docx
    doc = docx.Document(path)
    doc_name = os.path.basename(path)
    doc_id = doc_name.rsplit(".", 1)[0]
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text:
        return []
    return [{"doc_id": doc_id, "doc_name": doc_name, "page": 1, "text": text}]


def load_txt(path: str) -> list[dict]:
    doc_name = os.path.basename(path)
    doc_id = doc_name.rsplit(".", 1)[0]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    return [{"doc_id": doc_id, "doc_name": doc_name, "page": 1, "text": text}]


def load_md(path: str) -> list[dict]:
    return load_txt(path)


def load_any(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"不支持的文件类型: {ext}")