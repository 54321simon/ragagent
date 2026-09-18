"""Embedding 封装。用 paraphrase-multilingual-MiniLM-L12-v2（轻量多语言）。"""
from sentence_transformers import SentenceTransformer

_MODEL = None
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embedder(model_name: str = MODEL_NAME):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    # 多语言模型不需要指令前缀
    return get_embedder().encode([query], normalize_embeddings=True)[0].tolist()