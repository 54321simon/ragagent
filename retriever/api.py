"""检索层对外接口。Agent 和前端只调这个文件。"""
from typing import List
from common.schemas import RetrievedChunk
from retriever.loader import load_pdf
from retriever.chunker import chunk_pages
from retriever.store import add_chunks, query_chunks, delete_doc, list_docs, get_collection

# ---------- 基础接口 ----------

def ingest_pdf(path: str, chunk_size: int = 512, overlap: int = 100) -> int:
    """入库一篇 PDF，返回 chunk 数。"""
    pages = load_pdf(path)
    chunks = chunk_pages(pages, chunk_size, overlap)
    add_chunks(chunks)
    # 清 BM25 缓存，下次检索时重建
    global _BM25_CACHE
    _BM25_CACHE = None
    return len(chunks)


def retrieve(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """纯向量检索。"""
    return [RetrievedChunk(**r) for r in query_chunks(query, topk)]


def remove_doc(doc_id: str):
    delete_doc(doc_id)


def get_docs() -> list[dict]:
    return list_docs()


# ---------- 混合检索 ----------

from retriever.retrievers.bm25 import BM25Retriever
from retriever.retrievers.rrf import rrf_fusion

_BM25_CACHE = None


def _get_all_chunks() -> list[dict]:
    """从 Chroma 拉全部 chunks（用于构建 BM25）。"""
    col = get_collection()
    data = col.get(include=["documents", "metadatas"])
    out = []
    for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        out.append({
            "chunk_id": cid,
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page": meta["page"],
            "text": doc,
        })
    return out


def _get_bm25() -> BM25Retriever:
    global _BM25_CACHE
    if _BM25_CACHE is None:
        _BM25_CACHE = BM25Retriever(_get_all_chunks())
    return _BM25_CACHE


def retrieve_hybrid(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """混合检索：向量 + BM25 + RRF 融合。"""
    # 1. 向量检索（多召回一些，给 RRF 融合空间）
    vec_results = query_chunks(query, topk=topk * 2)
    # 2. BM25 检索
    bm25 = _get_bm25()
    bm25_results = bm25.search(query, topk=topk * 2)
    # 3. RRF 融合
    fused = rrf_fusion(vec_results, bm25_results, topk=topk)
    return [RetrievedChunk(
        chunk_id=x["chunk_id"],
        doc_id=x["doc_id"],
        doc_name=x["doc_name"],
        page=x["page"],
        text=x["text"],
        score=x.get("rrf_score", x.get("score", 0.0)),
    ) for x in fused]