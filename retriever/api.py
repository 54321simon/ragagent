"""检索层对外接口。Agent 和前端只调这个文件。"""
import os
from typing import List
from common.schemas import RetrievedChunk
from retriever.chunker import chunk_pages
from retriever.store import add_chunks, query_chunks, delete_doc, list_docs, get_collection

# ---------- 基础接口 ----------

def ingest_pdf(path: str, chunk_size: int = 1024, overlap: int = 200,
               chunk_mode: str = None) -> int:
    """入库一篇文档（PDF / DOCX / TXT / MD），返回 chunk 数。
    chunk_mode: 'page'（页内切分）/ 'section'（章节感知，默认）
    """
    from retriever.chunker import chunk_doc_by_section
    from retriever.loader import load_any

    if chunk_mode is None:
        chunk_mode = os.getenv("CHUNK_MODE", "section").lower()

    # load_any 自动识别 PDF / DOCX / TXT / MD
    pages = load_any(path)

    if chunk_mode == "page":
        chunks = chunk_pages(pages, chunk_size, overlap)
    else:
        chunks = chunk_doc_by_section(pages, chunk_size, overlap)

    add_chunks(chunks)
    global _BM25_CACHE
    _BM25_CACHE = None
    return len(chunks)


def retrieve(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """纯向量检索（bge-m3）。"""
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
    """混合检索：向量 + BM25 + RRF 融合。展示分用向量 cosine 分。"""
    vec_results = query_chunks(query, topk=topk * 2)
    bm25 = _get_bm25()
    bm25_results = bm25.search(query, topk=topk * 2)
    fused = rrf_fusion(vec_results, bm25_results, topk=topk)

    # RRF 只用于内部排序；展示分用向量 cosine 分，保证跨档可比
    vec_score_map = {r["chunk_id"]: r["score"] for r in vec_results}

    return [RetrievedChunk(
        chunk_id=x["chunk_id"],
        doc_id=x["doc_id"],
        doc_name=x["doc_name"],
        page=x["page"],
        text=x["text"],
        score=vec_score_map.get(x["chunk_id"], 0.0),
    ) for x in fused]


# ---------- 混合检索 + rerank ----------
from retriever.retrievers.reranker import rerank as _rerank


def retrieve_hybrid_rerank(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """
    三档检索最强版：向量 + BM25 → RRF → reranker。
    展示分用 reranker 的 sigmoid 分（0~1），和 vector/hybrid 档可比。
    """
    vec_results = query_chunks(query, topk=topk * 3)
    bm25 = _get_bm25()
    bm25_results = bm25.search(query, topk=topk * 3)
    fused = rrf_fusion(vec_results, bm25_results, topk=topk * 4)

    if not fused:
        return []

    reranked = _rerank(query, fused, topk=topk)

    return [RetrievedChunk(
        chunk_id=x["chunk_id"],
        doc_id=x["doc_id"],
        doc_name=x["doc_name"],
        page=x["page"],
        text=x["text"],
        score=x.get("rerank_score", x.get("rrf_score", 0.0)),
    ) for x in reranked]


# ---------- 统一入口（按环境变量决定走哪档） ----------

# vector（bge-m3 纯向量） / hybrid（+BM25+RRF，默认） / rerank（+bge-reranker）
_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()


def get_retrieval_mode() -> str:
    """返回当前检索模式，供 UI/日志显示。"""
    return _RETRIEVAL_MODE


def retrieve_best(query: str, topk: int = 5) -> List[RetrievedChunk]:
    """
    统一检索入口。默认 hybrid（bge-m3 + BM25 + RRF）。
    RETRIEVAL_MODE=vector  → 纯向量
    RETRIEVAL_MODE=rerank  → 向量 + BM25 + RRF + reranker
    """
    if _RETRIEVAL_MODE == "rerank":
        return retrieve_hybrid_rerank(query, topk)
    if _RETRIEVAL_MODE == "vector":
        return retrieve(query, topk)
    return retrieve_hybrid(query, topk)