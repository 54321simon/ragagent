"""Embedding 封装。用 bge-small-zh。"""
from sentence_transformers import SentenceTransformer

_MODEL = None


def get_embedder(model_name: str = "BAAI/bge-small-zh-v1.5"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    # bge 系列建议 query 加指令前缀
    return get_embedder().encode(
        [f"为这个句子生成表示以用于检索：{query}"],
        normalize_embeddings=True
    )[0].tolist()