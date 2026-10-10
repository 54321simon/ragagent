"""Multi-paper retrieval evaluation. Gold identity is (doc_id, physical PDF page)."""

import argparse, json, sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.api import retrieve, retrieve_hybrid, retrieve_hybrid_rerank


def score(retrieved, gold, k=5):
    gold = {(g["doc_id"], int(g["page"])) for g in gold}
    found = [(c.doc_id, int(c.page)) for c in retrieved[:k]]
    overlap = set(found) & gold
    return {
        "hit@1": float(bool(found and found[0] in gold)),
        "hit@5": float(bool(overlap)),
        "mrr@5": next((1 / (i + 1) for i, x in enumerate(found) if x in gold), 0),
        "recall@5": len(overlap) / len(gold) if gold else 0,
        "all_sources@5": float(gold.issubset(set(found))) if gold else 0,
    }


def run(modes, repeat=1):
    dataset = json.loads(
        (ROOT / "experiments/eval_set.json").read_text(encoding="utf-8")
    )
    output = {}
    for mode in modes:
        fn = {
            "vector": retrieve,
            "hybrid": retrieve_hybrid,
            "rerank": retrieve_hybrid_rerank,
            "hybrid_global": retrieve_hybrid,
        }[mode]
        rows = []
        for item in dataset:
            scope = None if mode == "hybrid_global" else (item["doc_ids"] or None)
            t0 = time.perf_counter()
            for _ in range(repeat):
                chunks = fn(item["question"], 5, scope)
            row = {
                "id": item["id"],
                "question": item["question"],
                "type": item["type"],
                "answerable": item["answerable"],
                "latency_ms": (time.perf_counter() - t0) * 1000 / repeat,
                "gold": item["gold_sources"],
                "top5": [
                    {
                        "doc_id": c.doc_id,
                        "page": c.page,
                        "score": c.score,
                        "chunk_id": c.chunk_id,
                        "text": c.text[:800],
                    }
                    for c in chunks
                ],
            }
            row.update(
                score(chunks, item["gold_sources"]) if item["answerable"] else {}
            )
            rows.append(row)
            print(
                mode,
                item["id"],
                round(row.get("hit@5", 0), 2),
                round(row["latency_ms"]),
                flush=True,
            )
        ans = [r for r in rows if r["answerable"]]
        times = [r["latency_ms"] for r in rows]
        summary = {
            k: statistics.mean(r[k] for r in ans)
            for k in ["hit@1", "hit@5", "mrr@5", "recall@5", "all_sources@5"]
        }
        summary.update(
            n_answerable=len(ans),
            n_unanswerable=len(rows) - len(ans),
            avg_latency_ms=statistics.mean(times),
            p95_latency_ms=sorted(times)[int(0.95 * (len(times) - 1))],
            score_type="reranker_sigmoid_penalized"
            if mode == "rerank"
            else "vector_cosine",
            repeat=repeat,
            scope="global"
            if mode == "hybrid_global"
            else "explicit question documents",
            query_embedding_cache="persistent; first query may be cold",
        )
        output[mode] = {"summary": summary, "rows": rows}
        path = ROOT / "experiments/results" / f"retrieval_{mode}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(output[mode], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["vector", "hybrid", "rerank"])
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()
    run(args.modes, args.repeat)
