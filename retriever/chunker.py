"""章节感知切分。先按章节聚合，再章节内按 size/overlap 切。
chunk 的 page 字段是「该 chunk 实际内容起始位置所在的页」。
"""
import re
from typing import List


# ---------- 基础切分（保留） ----------

def fixed_chunk(text: str, size: int = 1024, overlap: int = 200) -> List[str]:
    if size <= overlap:
        raise ValueError("size 必须大于 overlap")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_pages(pages: list[dict], size: int = 1024, overlap: int = 200) -> list[dict]:
    """把 [{doc_id, page, text}] 切成 [{chunk_id, doc_id, doc_name, page, text}]"""
    out = []
    for p in pages:
        for i, ch in enumerate(fixed_chunk(p["text"], size, overlap)):
            if ch.strip():
                out.append({
                    "chunk_id": f"{p['doc_id']}-p{p['page']}-c{i}",
                    "doc_id": p["doc_id"],
                    "doc_name": p["doc_name"],
                    "page": p["page"],
                    "text": ch,
                })
    return out


def recursive_chunk(text: str, size: int = 1024, overlap: int = 200) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_pages_recursive(pages: list[dict], size: int = 1024, overlap: int = 200) -> list[dict]:
    out = []
    for p in pages:
        for i, ch in enumerate(recursive_chunk(p["text"], size, overlap)):
            if ch.strip():
                out.append({
                    "chunk_id": f"{p['doc_id']}-p{p['page']}-r{i}",
                    "doc_id": p["doc_id"],
                    "doc_name": p["doc_name"],
                    "page": p["page"],
                    "text": ch,
                })
    return out


# ---------- 章节感知切分 ----------

SECTION_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s\-:,&()]{2,80})\s*$",
    re.MULTILINE,
)

SECTION_KEYWORDS = [
    "Introduction", "Background", "Related Work", "Method", "Approach",
    "Model", "Architecture", "Experiment", "Evaluation", "Result",
    "Discussion", "Conclusion", "Future Work", "References", "Abstract",
]


def _is_section_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if SECTION_PATTERN.match(line):
        return True
    if line in SECTION_KEYWORDS:
        return True
    return False


def split_into_sections(pages: list[dict]):
    """
    把所有页文本拼成全文，按章节标题切。
    返回：
      sections: [{section_title, start_pos, start_page, text}]
      page_offsets: [(char_pos, page_no), ...]
    """
    full_text = ""
    page_offsets = []
    for p in pages:
        page_offsets.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n"

    matches = []
    for m in SECTION_PATTERN.finditer(full_text):
        matches.append((m.start(), m.group(0).strip()))
    for kw in SECTION_KEYWORDS:
        for m in re.finditer(rf"^\s*{re.escape(kw)}\s*$", full_text, re.MULTILINE):
            pos = m.start()
            if not any(abs(pos - p) < 5 for p, _ in matches):
                matches.append((pos, kw))

    matches.sort(key=lambda x: x[0])

    if not matches:
        return [{
            "section_title": "full",
            "start_pos": 0,
            "start_page": page_offsets[0][1] if page_offsets else 1,
            "text": full_text,
        }], page_offsets

    sections = []
    if matches[0][0] > 0:
        sections.append({
            "section_title": "preamble",
            "start_pos": 0,
            "start_page": page_offsets[0][1] if page_offsets else 1,
            "text": full_text[:matches[0][0]],
        })

    for i, (pos, title) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        sections.append({
            "section_title": title,
            "start_pos": pos,
            "start_page": _page_at_pos(page_offsets, pos),
            "text": full_text[pos:end],
        })

    return sections, page_offsets


def _page_at_pos(page_offsets: list, pos: int) -> int:
    """给定字符位置，返回所在页码。"""
    if not page_offsets:
        return 1
    page = page_offsets[0][1]
    for start, pno in page_offsets:
        if start <= pos:
            page = pno
        else:
            break
    return page


def chunk_doc_by_section(pages: list[dict], size: int = 1024, overlap: int = 200) -> list[dict]:
    """
    章节感知切分。
    chunk 的 page 字段是「该 chunk 实际内容起始位置所在的页」（精确到 chunk）。
    """
    if not pages:
        return []

    doc_id = pages[0]["doc_id"]
    doc_name = pages[0]["doc_name"]

    sections, page_offsets = split_into_sections(pages)

    out = []
    for s_idx, sec in enumerate(sections):
        text = sec["text"]
        if not text.strip():
            continue

        sec_start = sec["start_pos"]
        step = size - overlap

        for c_idx, ch in enumerate(fixed_chunk(text, size, overlap)):
            if not ch.strip():
                continue
            # chunk 在全文里的绝对起始位置
            chunk_start_in_full = sec_start + c_idx * step
            page = _page_at_pos(page_offsets, chunk_start_in_full)

            out.append({
                "chunk_id": f"{doc_id}-p{page}-s{s_idx}-c{c_idx}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page,
                "section": sec["section_title"],
                "text": ch,
            })

    return out