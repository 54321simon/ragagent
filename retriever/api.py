"""Public retrieval API; scope filters precede ranking."""

import os, threading, time
from common.schemas import RetrievedChunk
from retriever.chunker import (
    chunk_pages,
    chunk_pages_recursive,
    chunk_pages_semantic,
    chunk_doc_by_section,
)
from retriever.store import (
    add_chunks,
    query_chunks,
    delete_doc,
    list_docs,
    get_collection,
    corpus_version,
)
from retriever.retrievers.bm25 import BM25Retriever
from retriever.retrievers.rrf import rrf_fusion
from retriever.retrievers.reranker import rerank

_BM25_CACHE = {}
_LOCK = threading.RLock()


def get_chunk_mode():
    return os.getenv("CHUNK_MODE", "section").lower()


def get_retrieval_mode():
    return os.getenv("RETRIEVAL_MODE", "hybrid").lower()


def ingest_pdf(path, chunk_size=1024, overlap=200, chunk_mode=None):
    from retriever.loader import load_any

    mode = chunk_mode or get_chunk_mode()
    fn = {
        "page": chunk_pages,
        "semantic": chunk_pages_semantic,
        "recursive": chunk_pages_recursive,
        "section": chunk_doc_by_section,
    }.get(mode)
    if fn is None:
        raise ValueError(f"未知切分方式 {mode}")
    chunks = fn(load_any(path), chunk_size, overlap)
    add_chunks(chunks)
    with _LOCK:
        _BM25_CACHE.clear()
    return len(chunks)


def ingest_with_retry(path, attempts=3, **kwargs):
    for i in range(attempts):
        try:
            return ingest_pdf(path, **kwargs)
        except (ValueError, FileNotFoundError):
            raise
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * 2**i)


def get_docs():
    return list_docs()


def remove_doc(doc_id):
    delete_doc(doc_id)
    with _LOCK:
        _BM25_CACHE.clear()


def _get_all_chunks(doc_ids=None):
    kw = {"where": {"doc_id": {"$in": list(doc_ids)}}} if doc_ids else {}
    d = get_collection().get(include=["documents", "metadatas"], **kw)
    return [
        dict(chunk_id=cid, text=t, **{k: m[k] for k in ("doc_id", "doc_name", "page")})
        for cid, t, m in zip(d["ids"], d["documents"], d["metadatas"])
    ]


def _get_bm25(doc_ids=None):
    key = (corpus_version(), tuple(sorted(doc_ids or [])))
    with _LOCK:
        if key not in _BM25_CACHE:
            if len(_BM25_CACHE) > 24:
                _BM25_CACHE.clear()
            _BM25_CACHE[key] = BM25Retriever(_get_all_chunks(doc_ids))
        return _BM25_CACHE[key]


def retrieve(query, topk=5, doc_ids=None):
    return [RetrievedChunk(**r) for r in query_chunks(query, topk, doc_ids)]


def retrieve_hybrid(query, topk=5, doc_ids=None):
    vec = query_chunks(query, topk * 3, doc_ids)
    bm = _get_bm25(doc_ids).search(query, topk * 3)
    fused = rrf_fusion(vec, bm, topk=topk, weight_a=0.7, weight_b=0.3)
    scores = {r["chunk_id"]: r["score"] for r in vec}
    return [
        RetrievedChunk(
            **{k: r[k] for k in ("chunk_id", "doc_id", "doc_name", "page", "text")},
            score=scores.get(r["chunk_id"], 0.0),
        )
        for r in fused
    ]


def retrieve_hybrid_rerank(query, topk=5, doc_ids=None):
    vec = query_chunks(query, topk * 3, doc_ids)
    bm = _get_bm25(doc_ids).search(query, topk * 3)
    fused = rrf_fusion(vec, bm, topk=topk * 4, weight_a=0.7, weight_b=0.3)
    return [
        RetrievedChunk(
            **{k: r[k] for k in ("chunk_id", "doc_id", "doc_name", "page", "text")},
            score=r["rerank_score"],
        )
        for r in rerank(query, fused, topk)
    ]


def retrieve_best(query, topk=5, doc_ids=None):
    return {
        "vector": retrieve,
        "hybrid": retrieve_hybrid,
        "rerank": retrieve_hybrid_rerank,
    }[get_retrieval_mode()](query, topk, doc_ids)
