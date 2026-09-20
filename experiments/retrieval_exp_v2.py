"""三档检索对比实验：纯向量 / 混合(RRF) / 混合+rerank。

判定方式：按页码命中。
额外统计：不可回答 query 的 top-1 分数（用于标定拒答阈值）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from retriever.api import retrieve, retrieve_hybrid, retrieve_hybrid_rerank


def hit_at_k(retrieved, relevant_pages, k=5):
    for item in retrieved[:k]:
        if item.page in relevant_pages:
            return 1.0
    return 0.0


def hit_at_1(retrieved, relevant_pages):
    return hit_at_k(retrieved, relevant_pages, k=1)


def mrr(retrieved, relevant_pages):
    for rank, item in enumerate(retrieved, start=1):
        if item.page in relevant_pages:
            return 1.0 / rank
    return 0.0


def _retrieve(mode, q, topk=5):
    if mode == "vector":
        return retrieve(q, topk=topk)
    if mode == "hybrid":
        return retrieve_hybrid(q, topk=topk)
    if mode == "hybrid_rerank":
        return retrieve_hybrid_rerank(q, topk=topk)
    raise ValueError(f"未知模式: {mode}")


def evaluate(eval_set, mode, topk=5, warmup=True, repeat=3):
    """跑评测。可回答 query 算 Hit@1/Hit@5/MRR，不可回答 query 只记录 top-1 分数。"""
    if warmup:
        _retrieve(mode, eval_set[0]["question"], topk=topk)

    answerable = [it for it in eval_set if it.get("answerable", True)]
    unanswerable = [it for it in eval_set if not it.get("answerable", True)]

    hits1, hits5, mrrs, times = [], [], [], []
    miss_details = []

    for item in answerable:
        q = item["question"]
        rel = set(item.get("relevant_pages", []))

        t0 = time.time()
        res = None
        for _ in range(repeat):
            res = _retrieve(mode, q, topk=topk)
        elapsed = (time.time() - t0) / repeat

        h1 = hit_at_1(res, rel)
        h5 = hit_at_k(res, rel, k=5)
        m = mrr(res, rel)

        hits1.append(h1)
        hits5.append(h5)
        mrrs.append(m)
        times.append(elapsed)

        if h5 == 0.0:
            miss_details.append({
                "question": q,
                "relevant_pages": sorted(rel),
                "top5": [
                    {"chunk_id": c.chunk_id, "page": c.page, "score": round(c.score, 3)}
                    for c in res[:5]
                ],
            })

    # 不可回答 query：只记录 top-1 分数
    unanswerable_scores = []
    for item in unanswerable:
        res = _retrieve(mode, item["question"], topk=topk)
        top1 = res[0].score if res else 0.0
        unanswerable_scores.append({
            "question": item["question"],
            "top1_score": round(top1, 3),
        })

    n = len(answerable)
    return {
        "mode": mode,
        "n_answerable": n,
        "n_unanswerable": len(unanswerable),
        "hit@1": sum(hits1) / n if n else 0,
        "hit@5": sum(hits5) / n if n else 0,
        "mrr": sum(mrrs) / n if n else 0,
        "avg_latency_ms": sum(times) / n * 1000 if n else 0,
        "miss_details": miss_details,
        "unanswerable_scores": unanswerable_scores,
    }


if __name__ == "__main__":
    with open("experiments/eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    n_ans = sum(1 for it in eval_set if it.get("answerable", True))
    n_unans = len(eval_set) - n_ans
    print(f"评测集 {len(eval_set)} 条（可回答 {n_ans} + 不可回答 {n_unans}）")
    print(f"warmup 后每档跑 3 次取平均\n")
    print(f"{'模式':<16} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} {'延迟(ms)':>10}")
    print("-" * 55)

    results = {}
    for mode in ["vector", "hybrid", "hybrid_rerank"]:
        r = evaluate(eval_set, mode)
        results[mode] = r
        print(f"{mode:<16} {r['hit@1']:>7.3f} {r['hit@5']:>7.3f} "
              f"{r['mrr']:>7.3f} {r['avg_latency_ms']:>10.0f}")

    # 未命中明细
    print("\n" + "=" * 60)
    print("未命中明细（可回答 query，Hit@5 = 0）")
    print("=" * 60)
    for mode, r in results.items():
        if not r["miss_details"]:
            print(f"\n【{mode}】全部命中 ✅")
            continue
        print(f"\n【{mode}】{len(r['miss_details'])} 条未命中：")
        for d in r["miss_details"]:
            print(f"  Q: {d['question']}")
            print(f"     gold pages: {d['relevant_pages']}")
            print(f"     top5: {[(c['page'], c['score']) for c in d['top5']]}")

    # 不可回答 query 的 top-1 分数
    print("\n" + "=" * 60)
    print("不可回答 query 的 top-1 分数（用于标定拒答阈值）")
    print("=" * 60)
    for mode, r in results.items():
        print(f"\n【{mode}】")
        for d in r["unanswerable_scores"]:
            print(f"  {d['top1_score']:.3f}  {d['question']}")