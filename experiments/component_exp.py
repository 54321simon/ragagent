"""Repeatable chunking and parser experiments, without modifying the live corpus."""

import sys, json, time, statistics
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.loader import load_pdf
from retriever.chunker import (
    chunk_pages,
    chunk_pages_recursive,
    chunk_pages_semantic,
    chunk_doc_by_section,
)
from retriever.embedder import embed_texts, embed_query
from common.schemas import RetrievedChunk
from experiments.evaluate import score


def main():
    data = json.loads((ROOT / "experiments/eval_set.json").read_text(encoding="utf-8"))
    questions = [
        d
        for d in data
        if d["id"] in ["F01", "F02", "F03", "F22", "F23", "F24", "R01", "R04"]
    ]
    pages = [
        p
        for name in ["Transformer", "MAE"]
        for p in load_pdf(str(ROOT / "data/papers" / f"{name}.pdf"))
    ]
    results = []
    configs = [
        ("fixed256", chunk_pages, 256, 50),
        ("fixed512", chunk_pages, 512, 100),
        ("fixed1024", chunk_pages, 1024, 200),
        ("recursive1024", chunk_pages_recursive, 1024, 200),
        ("semantic1024", chunk_pages_semantic, 1024, 200),
        ("section1024", chunk_doc_by_section, 1024, 200),
    ]
    for name, fn, size, overlap in configs:
        t0 = time.perf_counter()
        chunks = []
        for doc_id in ["Transformer", "MAE"]:
            chunks += fn([p for p in pages if p["doc_id"] == doc_id], size, overlap)
        split_s = time.perf_counter() - t0
        embed_start = time.perf_counter()
        vectors = np.array(embed_texts([c["text"] for c in chunks]))
        embed_s = time.perf_counter() - embed_start
        rows = []
        for item in questions:
            q = np.array(embed_query(item["question"]))
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
            strategy=name,
            chunk_size=size,
            overlap=overlap,
            docs=2,
            questions=len(questions),
            chunks=len(chunks),
            avg_length=statistics.mean(len(c["text"]) for c in chunks),
            min_length=min(len(c["text"]) for c in chunks),
            split_s=split_s,
            embedding_s=embed_s,
            **{
                key: statistics.mean(row[key] for row in rows)
                for key in ["hit@5", "mrr@5", "recall@5"]
            },
        )
        results.append({"summary": summary, "rows": rows})
        print(summary, flush=True)
        (ROOT / "experiments/results/chunking.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # Benchmark same 2 PDFs with three parsers. Extraction counts are diagnostic, not OCR accuracy.
    parsers = []
    for name in ["pypdf", "pdfplumber", "pymupdf"]:
        t0 = time.perf_counter()
        chars = pages_n = 0
        for file in ["Transformer.pdf", "MAE.pdf"]:
            path = ROOT / "data/papers" / file
            if name == "pypdf":
                from pypdf import PdfReader

                reader = PdfReader(path)
                texts = [p.extract_text() or "" for p in reader.pages]
            elif name == "pdfplumber":
                import pdfplumber

                with pdfplumber.open(path) as pdf:
                    texts = [p.extract_text() or "" for p in pdf.pages]
            else:
                import pymupdf

                with pymupdf.open(path) as pdf:
                    texts = [p.get_text() for p in pdf]
            pages_n += len(texts)
            chars += sum(map(len, texts))
        parsers.append(
            dict(
                parser=name,
                pages=pages_n,
                characters=chars,
                elapsed_s=time.perf_counter() - t0,
            )
        )
    (ROOT / "experiments/results/parsers.json").write_text(
        json.dumps(parsers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(parsers, flush=True)


if __name__ == "__main__":
    main()
