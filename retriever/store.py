"""Chroma persistence, scoped retrieval and versioned corpus mutations."""

import threading
import chromadb
from chromadb.config import Settings
from common.config import INDEX_DIR, EMBEDDING_MODEL
from retriever.embedder import embed_texts, embed_query

_LOCK = threading.RLock()
_CLIENT = None
_COLLECTION = None
COLLECTION_NAME = "papers"


def get_collection(persist_dir=None):
    global _CLIENT, _COLLECTION
    with _LOCK:
        if _COLLECTION is None:
            _CLIENT = chromadb.PersistentClient(
                path=str(persist_dir or INDEX_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            _COLLECTION = _CLIENT.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
            )
            model = (_COLLECTION.metadata or {}).get("embedding_model")
            if model and model != EMBEDDING_MODEL:
                raise RuntimeError(
                    "Embedding 模型已改变，请使用新 INDEX_DIR 重新入库。"
                )
        return _COLLECTION


def corpus_version():
    return str((get_collection().metadata or {}).get("revision", "0"))


def _bump(col):
    import uuid

    metadata = {k: v for k, v in (col.metadata or {}).items() if k != "hnsw:space"}
    metadata["revision"] = uuid.uuid4().hex
    col.modify(metadata=metadata)


def add_chunks(chunks):
    if not chunks:
        raise ValueError("未提取到有效文本，扫描版 PDF 需要先做 OCR。")
    vectors = embed_texts([c["text"] for c in chunks])
    col = get_collection()
    with _LOCK:
        old = col.get(where={"doc_id": chunks[0]["doc_id"]}, include=[])["ids"]
        col.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {k: c.get(k, "") for k in ("doc_id", "doc_name", "page", "section")}
                for c in chunks
            ],
        )
        obsolete = set(old) - {c["chunk_id"] for c in chunks}
        if obsolete:
            col.delete(ids=list(obsolete))
        _bump(col)


def query_chunks(query, topk=5, doc_ids=None):
    if topk <= 0:
        return []
    col = get_collection()
    if not col.count():
        return []
    where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
    if where and not col.get(where=where, include=[], limit=1)["ids"]:
        return []
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=min(topk, col.count()),
        **({"where": where} if where else {}),
    )
    return [
        dict(
            chunk_id=cid,
            text=doc,
            doc_id=m["doc_id"],
            doc_name=m["doc_name"],
            page=m["page"],
            score=1 - float(dist),
        )
        for cid, doc, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def list_docs():
    data = get_collection().get(include=["metadatas"])
    seen = {}
    for m in data["metadatas"]:
        d = seen.setdefault(
            m["doc_id"], dict(doc_id=m["doc_id"], doc_name=m["doc_name"], chunks=0)
        )
        d["chunks"] += 1
    return sorted(seen.values(), key=lambda x: x["doc_id"])


def delete_doc(doc_id):
    with _LOCK:
        col = get_collection()
        col.delete(where={"doc_id": doc_id})
        _bump(col)
