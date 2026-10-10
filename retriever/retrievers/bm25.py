"""BM25 关键词检索。用 rank_bm25 + jieba 中文分词。"""

import jieba
import re
from common.config import DATA_DIR

_jieba_cache = DATA_DIR / "cache"
_jieba_cache.mkdir(parents=True, exist_ok=True)
jieba.dt.tmp_dir = str(_jieba_cache)


def tokenize(text):
    return [t.lower() for t in jieba.cut(text) if re.search(r"[\w\u4e00-\u9fff]", t)]


from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        """chunks: [{chunk_id, doc_id, doc_name, page, text}]"""
        self.chunks = chunks
        # 中文分词
        self.tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def search(self, query: str, topk: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # 取 topk 索引
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
        return [
            {**self.chunks[i], "score": float(scores[i])} for i in idx if scores[i] > 0
        ]
