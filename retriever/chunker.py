"""固定大小切分。后面再加递归、语义切分。"""
from typing import List


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
    """递归字符切分。优先按段落/句子切，避免把句子切两半。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_pages_recursive(pages: list[dict], size: int = 1024, overlap: int = 200) -> list[dict]:
    """用递归切分处理 pages。输出格式与 chunk_pages 一致。"""
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