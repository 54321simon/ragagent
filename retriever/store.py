"""Chroma 向量库封装。持久化到 data/index/。"""
import os
import chromadb
from retriever.embedder import embed_texts, embed_query

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_INDEX = os.path.join(_BASE, "data", "index")

_CLIENT = None
_COLLECTION = None
COLLECTION_NAME = "papers"


def get_collection(persist_dir: str = _DEFAULT_INDEX):
    global _CLIENT, _COLLECTION
    if _COLLECTION is None:
        _CLIENT = chromadb.PersistentClient(path=persist_dir)
        _COLLECTION = _CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def add_chunks(chunks: list[dict]):
    col = get_collection()
    texts = [c["text"] for c in chunks]
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embed_texts(texts),
        documents=texts,
        metadatas=[{
            "doc_id": c["doc_id"],
            "doc_name": c["doc_name"],
            "page": c["page"],
        } for c in chunks],
    )


def query_chunks(query: str, topk: int = 5) -> list[dict]:
    col = get_collection()
    res = col.query(query_embeddings=[embed_query(query)], n_results=topk)
    out = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0],
    ):
        out.append({
            "chunk_id": cid,
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page": meta["page"],
            "text": doc,
            "score": 1 - dist,
        })
    return out


def list_docs() -> list[dict]:
    col = get_collection()
    data = col.get(include=["metadatas"])
    seen = {}
    for m in data["metadatas"]:
        seen[m["doc_id"]] = m["doc_name"]
    return [{"doc_id": k, "doc_name": v} for k, v in seen.items()]


def delete_doc(doc_id: str):
    get_collection().delete(where={"doc_id": doc_id})