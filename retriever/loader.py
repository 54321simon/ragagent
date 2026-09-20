"""PDF 加载器。用 pymupdf。自动过滤页眉/页脚/版权声明/参考文献页。"""
import pymupdf
import os
import re
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


BODY_KEYWORDS = [
    "conclusion", "discussion", "future work", "acknowledg",
    "introduction", "method", "approach", "experiment", "result",
    "abstract", "background",
]

REF_KEYWORDS = [
    "arxiv:", "corr,", "in proc.", "proceedings",
    "et al.", "conference on", "journal of", "preprint",
]


def _is_reference_page(text: str) -> bool:
    """判断是否是纯参考文献页。
    判据：1) [数字] 行占比 > 0.25；2) 参考文献特征词 >= 3 个；
         3) 页首出现 References 标题。
    含正文关键词的页（Conclusion 等）永远不判为参考文献页。
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return False

    # 0. 正文关键词豁免：p10 含 "Table 4" + "Conclusion"，直接保留
    low_text = text.lower()
    # 只有明确含 Conclusion/Discussion/Future Work 才豁免
    # 不含 method/approach/result 等泛词，避免参考文献误豁免
    strong_body_kw = ["conclusion", "discussion", "future work",
                      "acknowledg", "introduction"]
    if any(kw in low_text for kw in strong_body_kw):
        return False

    # 1. 强特征：[数字] 行占比 > 0.25
    ref_pattern = re.compile(r"^\[\d+(?:\s*[,-]\s*\d+)*\]")
    ref_lines = sum(1 for l in lines if ref_pattern.match(l))
    if ref_lines / len(lines) > 0.25:
        return True

    # 2. 辅助特征：参考文献常见词 >= 3 个
    kw_hits = sum(1 for kw in REF_KEYWORDS if kw in low_text)
    if kw_hits >= 3:
        return True

    # 3. References 标题只在页首 3 行内才算
    head = "\n".join(lines[:3])
    if re.search(r"^\s*References\s*$", head, re.M | re.I):
        return True

    return False


def load_pdf(path: str) -> list[dict]:
    """返回 [{doc_id, doc_name, page, text}]。
    自动过滤：页眉/页脚 + 版权声明 + 参考文献页。"""
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

    # 过滤 2：版权 + 参考文献页
    pages = []
    for p in raw_pages:
        # 参考文献页整页跳过
        if _is_reference_page(p["text"]):
            continue

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