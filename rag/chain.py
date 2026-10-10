"""RAG generation, evidence logs, semantic cache and three fallback classes."""

import os, time
from dataclasses import asdict
from common import llm as ollama
from common.config import MODEL
from common.observability import write_log
from common.schemas import RetrievedChunk
from retriever.api import retrieve_best, get_retrieval_mode
from retriever.store import corpus_version
from retriever.embedder import embed_query
from rag.cache import SemanticCache, fingerprint
from rag.prompts import RAG_SYSTEM_PROMPT
from rag.context import build_context
from rag.citation import extract_citations, validate_citations
from rag.fallback import handle_no_results, handle_low_relevance, handle_llm_error

DEFAULT_MODEL = MODEL
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
LOW_RELEVANCE_THRESHOLD = float(os.getenv("LOW_RELEVANCE_THRESHOLD", "0.01"))


def get_evidence(query, topk=5, doc_ids=None):
    t0 = time.perf_counter()
    chunks = retrieve_best(query, topk, doc_ids)
    write_log(
        "retrieval",
        dict(
            question=query,
            doc_ids=doc_ids,
            mode=get_retrieval_mode(),
            score_type="reranker_sigmoid_penalized"
            if get_retrieval_mode() == "rerank"
            else "vector_cosine",
            top1_score=chunks[0].score if chunks else None,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            retrieved=[asdict(c) for c in chunks],
        ),
    )
    return chunks


def _prepare(query, topk=5, use_hybrid=True, doc_ids=None):
    chunks = get_evidence(query, topk, doc_ids)
    if not chunks:
        return [], None, handle_no_results(query)
    if max(c.score for c in chunks) < LOW_RELEVANCE_THRESHOLD:
        return (
            chunks,
            None,
            handle_low_relevance(query, chunks, LOW_RELEVANCE_THRESHOLD),
        )
    if os.getenv("SELF_CHECK_ENABLED", "1") == "1":
        from rag.self_check import check_answerable

        if not check_answerable(query, chunks):
            return (
                chunks,
                None,
                dict(
                    answer="知识库中未找到能回答该问题的内容，请换个问法或上传相关论文。",
                    citations=[],
                    degraded=True,
                    refused=True,
                ),
            )
    return (
        chunks,
        RAG_SYSTEM_PROMPT.format(context=build_context(chunks), question=query),
        None,
    )


def _scope(model, topk, doc_ids, options):
    return fingerprint(
        [
            "rag-v2",
            model,
            topk,
            doc_ids or [],
            corpus_version(),
            get_retrieval_mode(),
            options,
        ]
    )


def rag_answer(
    query, topk=5, model=DEFAULT_MODEL, doc_ids=None, options=None, use_cache=True
):
    t0 = time.perf_counter()
    cache = SemanticCache()
    scope = _scope(model, topk, doc_ids, options)
    vector = None
    if use_cache and os.getenv("CACHE_ENABLED", "1") == "1":
        cached, similarity = cache.get(query, scope)
        if not cached:
            try:
                vector = embed_query(query)
                cached, similarity = cache.get(query, scope, vector)
            except Exception:
                cached = None
        if cached:
            cached["citations"] = [RetrievedChunk(**c) for c in cached["citations"]]
            cached["metrics"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "tokens": 0,
                "model": model,
                "cache_hit": True,
                "cache_similarity": similarity,
            }
            write_log(
                "rag",
                dict(
                    question=query,
                    **{k: v for k, v in cached.items() if k != "citations"},
                ),
            )
            return cached
    try:
        chunks, prompt, degraded = _prepare(query, topk, doc_ids=doc_ids)
        if degraded:
            if degraded.get("degrade_type") == "no_results":
                try:
                    r = ollama.chat(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"当前知识库中未找到相关文档。可用一般知识回答，但不得伪造任何论文引用，注明未经过知识库核验。问题：{query}",
                            }
                        ],
                        options={"temperature": 0.1, "num_predict": MAX_TOKENS},
                    )
                    degraded["answer"] = (
                        "当前知识库中未找到相关文档。以下为模型一般知识回答，未经论文核验。\n\n"
                        + r["message"]["content"]
                    )
                    degraded["metrics"] = {
                        "tokens": r.get("prompt_eval_count", 0) + r.get("eval_count", 0)
                    }
                except Exception as e:
                    degraded = handle_llm_error(query, str(e))
            degraded.setdefault("metrics", {}).update(
                elapsed_ms=round((time.perf_counter() - t0) * 1000),
                model=model,
                cache_hit=False,
            )
            write_log(
                "rag",
                dict(
                    question=query,
                    answer=degraded["answer"],
                    metrics=degraded["metrics"],
                    degraded=True,
                ),
            )
            return degraded
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": MAX_TOKENS, **(options or {})},
        )
    except Exception as e:
        return handle_llm_error(query, str(e))
    answer = resp["message"]["content"]
    cited = extract_citations(answer, chunks)
    debug = validate_citations(answer, chunks)
    result = dict(
        answer=answer,
        citations=cited,
        metrics=dict(
            elapsed_ms=round((time.perf_counter() - t0) * 1000),
            tokens=resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0),
            model=model,
            cache_hit=False,
        ),
        degraded=False,
        refused=False,
        debug=debug,
    )
    allowed = {c.cite() for c in chunks}
    import re

    markers = re.findall(r"【[^】]+?-第\d+页】", answer)
    if use_cache and markers and all(m in allowed for m in markers):
        try:
            cache.put(
                query,
                scope,
                vector or embed_query(query),
                {**result, "citations": [asdict(c) for c in cited]},
            )
        except Exception:
            pass
    write_log(
        "rag",
        dict(
            question=query,
            answer=answer,
            citations=[asdict(c) for c in cited],
            metrics=result["metrics"],
            debug=debug,
        ),
    )
    return result


def rag_answer_stream(query, topk=5, model=DEFAULT_MODEL, doc_ids=None):
    chunks, prompt, degraded = _prepare(query, topk, doc_ids=doc_ids)
    if degraded:
        yield degraded["answer"]
        return
    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.1, "num_predict": MAX_TOKENS},
        ):
            yield chunk["message"]["content"]
    except Exception as e:
        yield handle_llm_error(query, str(e))["answer"]


def rag_answer_with_citations(query, topk=5, model=DEFAULT_MODEL):
    chunks, prompt, degraded = _prepare(query, topk)
    if degraded:
        return degraded, None
    collected = []

    def gen():
        for piece in rag_answer_stream(query, topk, model):
            collected.append(piece)
            yield piece

    def finalize():
        answer = "".join(collected)
        return dict(
            answer=answer, citations=extract_citations(answer, chunks), degraded=False
        )

    return gen, finalize
