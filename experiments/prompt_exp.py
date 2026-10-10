"""Generation-parameter ablation using exactly the same retrieved evidence."""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import llm
from common.config import MODEL
from retriever.api import retrieve
from rag.context import build_context
from rag.prompts import RAG_SYSTEM_PROMPT
from rag.citation import validate_citations


def main():
    q = "MAE 的 asymmetric encoder-decoder 如何工作？"
    chunks = retrieve(q, 5, ["MAE"])
    prompt = RAG_SYSTEM_PROMPT.format(context=build_context(chunks), question=q)
    rows = []
    for opts in [
        {"temperature": 0.0, "top_p": 0.9, "top_k": 40},
        {"temperature": 0.5, "top_p": 0.9, "top_k": 40},
        {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
        {"temperature": 0.1, "top_p": 0.7, "top_k": 20},
    ]:
        t0 = time.perf_counter()
        resp = llm.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**opts, "num_predict": 300},
        )
        answer = resp["message"]["content"]
        row = dict(
            options=opts,
            question=q,
            answer=answer,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            completion_tokens=resp.get("eval_count", 0),
            citation_validation=validate_citations(answer, chunks),
            human_quality_score=None,
        )
        rows.append(row)
        print(opts, row["elapsed_ms"], flush=True)
        (ROOT / "experiments/results/generation_parameters.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
