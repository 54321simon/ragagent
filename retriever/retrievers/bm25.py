"""BM25 关键词检索。用 rank_bm25 + jieba 中文分词。"""
import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        """chunks: [{chunk_id, doc_id, doc_name, page, text}]"""
        self.chunks = chunks
        # 中文分词
        self.tokenized = [list(jieba.cut(c["text"])) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def search(self, query: str, topk: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        # 取 topk 索引
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in idx if scores[i] > 0
        ]