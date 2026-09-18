"""三档检索对比实验：纯向量 / 混合(RRF) / 混合+rerank。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from retriever.api import retrieve, retrieve_hybrid, retrieve_hybrid_rerank
import json
import time
from retriever.api import retrieve, retrieve_hybrid, retrieve_hybrid_rerank


def hit_at_k(retrieved: list, relevant: set, k: int = 5) -> float:
    for item in retrieved[:k]:
        cid = item.chunk_id if hasattr(item, "chunk_id") else item["chunk_id"]
        if cid in relevant:
            return 1.0
    return 0.0


def mrr(retrieved: list, relevant: set) -> float:
    for rank, item in enumerate(retrieved, start=1):
        cid = item.chunk_id if hasattr(item, "chunk_id") else item["chunk_id"]
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(eval_set: list[dict], mode: str = "vector") -> dict:
    hits, mrrs, times = [], [], []
    for item in eval_set:
        q = item["question"]
        rel = set(item["relevant_chunk_ids"])

        t0 = time.time()
        if mode == "vector":
            res = retrieve(q, topk=5)
        elif mode == "hybrid":
            res = retrieve_hybrid(q, topk=5)
        elif mode == "hybrid_rerank":
            res = retrieve_hybrid_rerank(q, topk=5)
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
    print(f"{'模式':<16} {'Hit@5':>8} {'MRR':>8} {'延迟(ms)':>12}")
    print("-" * 50)
    for mode in ["vector", "hybrid", "hybrid_rerank"]:
        r = evaluate(eval_set, mode)
        print(f"{mode:<16} {r['hit@5']:>8.3f} {r['mrr']:>8.3f} {r['avg_latency_ms']:>12.0f}")