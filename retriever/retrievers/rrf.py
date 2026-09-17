"""RRF 倒数排名融合。手写实现，不调库。"""
from typing import List


def rrf_fusion(
    results_a: List[dict],
    results_b: List[dict],
    k: int = 60,
    topk: int = 5,
) -> List[dict]:
    """
    融合两路检索结果。
    score = 1/(k + rank_a) + 1/(k + rank_b)
    两路都用 chunk_id 去重。
    """
    scores = {}
    meta = {}

    for rank, item in enumerate(results_a):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        meta[cid] = item

    for rank, item in enumerate(results_b):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        meta[cid] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:topk]
    return [{**meta[cid], "rrf_score": scores[cid]} for cid in sorted_ids]