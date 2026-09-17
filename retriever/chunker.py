"""固定大小切分。后面再加递归、语义切分。"""
from typing import List


def fixed_chunk(text: str, size: int = 512, overlap: int = 50) -> List[str]:
    if size <= overlap:
        raise ValueError("size 必须大于 overlap")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_pages(pages: list[dict], size: int = 512, overlap: int = 50) -> list[dict]:
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