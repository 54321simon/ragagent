"""Three embedding models on identical two-paper chunks and eight questions."""

import sys, json, time, statistics, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.loader import load_pdf
from retriever.chunker import chunk_doc_by_section
from retriever.embedder import embed_texts
from common.schemas import RetrievedChunk
from experiments.evaluate import score


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    data = json.loads((ROOT / "experiments/eval_set.json").read_text(encoding="utf-8"))
    questions = [
        d
        for d in data
        if d["id"] in ["F01", "F02", "F03", "F22", "F23", "F24", "R01", "R04"]
    ]
    chunks = []
    for name in ["Transformer", "MAE"]:
        chunks += chunk_doc_by_section(
            load_pdf(str(ROOT / "data/papers" / f"{name}.pdf")), 1024, 200
        )
    results = []
    for name in ["bge-m3:567m", "m3e-base", "bge-large-zh-v1.5"]:
        start = time.perf_counter()
        if name == "bge-m3:567m":
            vectors = np.array(embed_texts([c["text"] for c in chunks]))
            queries = np.array(embed_texts([q["question"] for q in questions]))
        else:
            model = SentenceTransformer(
                str(ROOT.parent / "models" / name), device="cpu", local_files_only=True
            )
            vectors = model.encode(
                [c["text"] for c in chunks],
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            prefix = (
                "为这个句子生成表示以用于检索相关文章："
                if name.startswith("bge-large")
                else ""
            )
            queries = model.encode(
                [prefix + q["question"] for q in questions],
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            del model
        rows = []
        for item, q in zip(questions, queries):
            eligible = [
                i for i, c in enumerate(chunks) if c["doc_id"] in item["doc_ids"]
            ]
            indices = sorted(eligible, key=lambda i: -float(vectors[i] @ q))[:5]
            retrieved = [
                RetrievedChunk(
                    **{
                        k: chunks[i][k]
                        for k in ["chunk_id", "doc_id", "doc_name", "page", "text"]
                    },
                    score=float(vectors[i] @ q),
                )
                for i in indices
            ]
            rows.append(
                {
                    "id": item["id"],
                    **score(retrieved, item["gold_sources"]),
                    "top5": [(c.doc_id, c.page) for c in retrieved],
                }
            )
        summary = dict(
            model=name,
            dimensions=int(vectors.shape[1]),
            docs=2,
            questions=8,
            chunks=len(chunks),
            elapsed_s=time.perf_counter() - start,
            **{
                k: statistics.mean(r[k] for r in rows)
                for k in ["hit@5", "mrr@5", "recall@5"]
            },
        )
        results.append({"summary": summary, "rows": rows})
        print(summary, flush=True)
        (ROOT / "experiments/results/embeddings.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
