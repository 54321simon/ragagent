"""bge-reranker 精排。用 CrossEncoder 对候选做细粒度打分。"""
from typing import List

_RERANKER = None
MODEL_NAME = "BAAI/bge-reranker-base"


def get_reranker():
    """懒加载 reranker 模型（首次会下载 ~1.1GB）。"""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(MODEL_NAME, max_length=512)
    return _RERANKER


def rerank(
    query: str,
    candidates: List[dict],
    topk: int = 5,
) -> List[dict]:
    """对候选 chunk 做精排。加长度惩罚，避免短文本霸榜。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["text"][:512]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    for c, s in zip(candidates, scores):
        # 长度惩罚：< 100 字 × 0.8，< 50 字 × 0.5
        text_len = len(c["text"])
        if text_len < 50:
            penalty = 0.5
        elif text_len < 100:
            penalty = 0.8
        else:
            penalty = 1.0
        c["rerank_score"] = float(s) * penalty

    sorted_cands = sorted(candidates, key=lambda x: -x["rerank_score"])
    return sorted_cands[:topk]