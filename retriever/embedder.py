"""Ollama embeddings with persistent cache and bounded retry."""

import hashlib, json, math, sqlite3, threading, time
from common import llm as ollama
from common.config import DATA_DIR, EMBEDDING_MODEL

MODEL_NAME = EMBEDDING_MODEL
_LOCK = threading.RLock()
_BATCH_SIZE = 16
_MAX_RETRY = 3


def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed_batch(texts):
    for attempt in range(_MAX_RETRY):
        try:
            resp = ollama.embed(model=MODEL_NAME, input=texts)
            if len(resp["embeddings"]) != len(texts):
                raise ValueError("Embedding 返回数量不匹配")
            return resp["embeddings"]
        except Exception as e:
            if attempt == _MAX_RETRY - 1:
                raise RuntimeError(f"Ollama embed 重试失败: {e}") from e
            time.sleep(0.3 * 2**attempt)


def _connection():
    con = sqlite3.connect(str(DATA_DIR / "embedding_cache.sqlite3"), timeout=60)
    con.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY,vector TEXT NOT NULL)"
    )
    return con


def embed_texts(texts):
    texts = [texts] if isinstance(texts, str) else texts
    if not texts:
        return []
    keys = [hashlib.sha256((MODEL_NAME + "\0" + t).encode()).hexdigest() for t in texts]
    with _LOCK, _connection() as con:
        cached = {
            k: json.loads(row[0])
            for k in set(keys)
            if (
                row := con.execute(
                    "SELECT vector FROM embeddings WHERE key=?", (k,)
                ).fetchone()
            )
        }
    pairs = list({k: t for k, t in zip(keys, texts) if k not in cached}.items())
    for i in range(0, len(pairs), _BATCH_SIZE):
        batch = pairs[i : i + _BATCH_SIZE]
        vectors = [_l2_normalize(v) for v in _embed_batch([t for _, t in batch])]
        with _LOCK, _connection() as con:
            con.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?)",
                [(k, json.dumps(v)) for (k, _), v in zip(batch, vectors)],
            )
        cached.update({k: v for (k, _), v in zip(batch, vectors)})
    return [cached[k] for k in keys]


def embed_query(query):
    return embed_texts([query])[0]
