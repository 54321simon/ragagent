"""bge-reranker 精排。加 sigmoid 归一化 + 长度惩罚 + 标题页惩罚。"""
import math
import re
from typing import List

_RERANKER = None
MODEL_NAME = "BAAI/bge-reranker-base"


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(MODEL_NAME, max_length=512)
    return _RERANKER


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, candidates: List[dict], topk: int = 5) -> List[dict]:
    """对候选 chunk 做精排。返回 score 为 sigmoid 后的 0~1 概率。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["text"][:512]) for c in candidates]
    logits = model.predict(pairs, show_progress_bar=False)

    for c, s in zip(candidates, logits):
        penalty = 1.0

        # 长度惩罚
        text_len = len(c["text"])
        if text_len < 50:
            penalty *= 0.5
        elif text_len < 100:
            penalty *= 0.8

        # 标题页惩罚：page 1 且 query 不显式问标题/作者
        if c.get("page") == 1:
            title_kw = ["标题", "题目", "作者", "是谁写的", "title"]
            if not any(kw in query.lower() for kw in title_kw):
                penalty *= 0.5

        # 参考文献惩罚（放宽正则）
        ref_lines = sum(
            1 for l in c["text"].split("\n")
            if re.match(r"^\[\d+(\s*,\s*\d+)*\]", l.strip())
        )
        if ref_lines >= 2:
            penalty *= 0.3

        c["rerank_score"] = _sigmoid(float(s)) * penalty

    sorted_cands = sorted(candidates, key=lambda x: -x["rerank_score"])
    return sorted_cands[:topk]