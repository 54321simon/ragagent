"""bge-reranker 精排。加长度惩罚 + 参考文献惩罚。"""
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


def rerank(query: str, candidates: List[dict], topk: int = 5) -> List[dict]:
    """对候选 chunk 做精排。加长度惩罚 + 参考文献惩罚。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["text"][:512]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    for c, s in zip(candidates, scores):
        penalty = 1.0

        # 长度惩罚：< 50 字 × 0.5，< 100 字 × 0.8
        text_len = len(c["text"])
        if text_len < 50:
            penalty *= 0.5
        elif text_len < 100:
            penalty *= 0.8

        # 参考文献惩罚：[数字] 开头的行 >= 2 时降权
        ref_lines = sum(
            1 for l in c["text"].split("\n")
            if re.match(r"^\[\d+\]", l.strip())
        )
        if ref_lines >= 2:
            penalty *= 0.3

        c["rerank_score"] = float(s) * penalty

    sorted_cands = sorted(candidates, key=lambda x: -x["rerank_score"])
    return sorted_cands[:topk]