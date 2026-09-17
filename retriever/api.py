"""检索层对外接口。Agent 和前端只调这个文件。"""
from typing import List
from common.schemas import RetrievedChunk
from retriever.loader import load_pdf
from retriever.chunker import chunk_pages
from retriever.store import add_chunks, query_chunks, delete_doc, list_docs


def ingest_pdf(path: str, chunk_size: int = 512, overlap: int = 50) -> int:
    """入库一篇 PDF，返回 chunk 数。"""
    pages = load_pdf(path)
    chunks = chunk_pages(pages, chunk_size, overlap)
    add_chunks(chunks)
    return len(chunks)


def retrieve(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """检索接口。签名与 common/interfaces.py 一致。"""
    return [RetrievedChunk(**r) for r in query_chunks(query, topk)]


def remove_doc(doc_id: str):
    delete_doc(doc_id)


def get_docs() -> list[dict]:
    return list_docs()