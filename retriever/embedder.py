"""Embedding 封装。用 Ollama 本地的 bge-m3:567m（1024 维）。"""
import math
import time
import ollama

MODEL_NAME = "bge-m3:567m"
_BATCH_SIZE = 32
_MAX_RETRY = 3


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    last_err = None
    for attempt in range(_MAX_RETRY):
        try:
            resp = ollama.embed(model=MODEL_NAME, input=texts)
            return resp["embeddings"]
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Ollama embed 失败（重试 {_MAX_RETRY} 次）: {last_err}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        out.extend(_embed_batch(batch))
    return [_l2_normalize(v) for v in out]


def embed_query(query: str) -> list[float]:
    return _l2_normalize(_embed_batch([query])[0])