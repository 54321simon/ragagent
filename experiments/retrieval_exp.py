"""三档检索对比：纯向量 / 混合(RRF)。
需要评测集：experiments/eval_set.json。
格式：[{"question": "...", "relevant_chunk_ids": ["..."]}, ...]
"""
import json
import time
from retriever.api import retrieve, retrieve_hybrid
from retriever.store import query_chunks


def hit_at_k(retrieved: list, relevant: set, k: int = 5) -> float:
    """前 k 个里有没有命中相关 chunk。"""
    top = retrieved[:k]
    for item in top:
        cid = item.chunk_id if hasattr(item, "chunk_id") else item["chunk_id"]
        if cid in relevant:
            return 1.0
    return 0.0


def mrr(retrieved: list, relevant: set) -> float:
    """第一个命中位置的倒数。"""
    for rank, item in enumerate(retrieved, start=1):
        cid = item.chunk_id if hasattr(item, "chunk_id") else item["chunk_id"]
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(eval_set: list[dict], mode: str = "vector") -> dict:
    """mode: vector / hybrid"""
    hits, mrrs, times = [], [], []
    for item in eval_set:
        q = item["question"]
        rel = set(item["relevant_chunk_ids"])

        t0 = time.time()
        if mode == "vector":
            res = retrieve(q, topk=5)
        elif mode == "hybrid":
            res = retrieve_hybrid(q, topk=5)
        else:
            raise ValueError(f"未知模式: {mode}")
        elapsed = time.time() - t0

        hits.append(hit_at_k(res, rel, k=5))
        mrrs.append(mrr(res, rel))
        times.append(elapsed)

    return {
        "mode": mode,
        "hit@5": sum(hits) / len(hits) if hits else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
        "avg_latency_ms": sum(times) / len(times) * 1000 if times else 0,
        "n": len(eval_set),
    }


if __name__ == "__main__":
    with open("experiments/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"评测集 {len(eval_set)} 条\n")
    for mode in ["vector", "hybrid"]:
        r = evaluate(eval_set, mode)
        print(f"{mode:10s}  Hit@5={r['hit@5']:.3f}  MRR={r['mrr']:.3f}  延迟={r['avg_latency_ms']:.0f}ms")