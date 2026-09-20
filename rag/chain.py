"""RAG 生成链。同步 + 流式两种模式。"""
import time
from typing import Iterator
import ollama

from common.schemas import RetrievedChunk
from retriever.api import retrieve_best
from rag.prompts import RAG_SYSTEM_PROMPT
from rag.context import build_context
from rag.citation import extract_citations, validate_citations
from rag.fallback import (
    handle_no_results,
    handle_low_relevance,
    handle_llm_error,
)

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
LOW_RELEVANCE_THRESHOLD = 0.01
MAX_TOKENS = 1024


def _handle_refuse(query: str, chunks: list) -> dict:
    """自检不通过时的拒答响应。"""
    return {
        "answer": "知识库中未找到能回答该问题的内容。请尝试换个问法，或上传相关论文。",
        "citations": [],
        "metrics": {"elapsed_ms": 0, "tokens": 0, "model": "self_check"},
        "degraded": True,
        "refused": True,
    }


def _prepare(query: str, topk: int = 5, use_hybrid: bool = True):
    """统一预处理：检索 + 判断降级 + 自检 + 拼接上下文。"""
    chunks = retrieve_best(query, topk=topk)

    if not chunks:
        return None, None, handle_no_results(query)

    best = max(c.score for c in chunks)
    if best < LOW_RELEVANCE_THRESHOLD:
        return chunks, None, handle_low_relevance(query, chunks)

    # LLM 自检：判断 chunk 能否回答 query
    from rag.self_check import check_answerable
    if not check_answerable(query, chunks):
        return chunks, None, _handle_refuse(query, chunks)

    context = build_context(chunks)
    prompt = RAG_SYSTEM_PROMPT.format(context=context, question=query)
    return chunks, prompt, None


def rag_answer(
    query: str,
    topk: int = 5,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    同步 RAG 问答。返回：
    {
        "answer": str,
        "citations": List[RetrievedChunk],
        "metrics": {"elapsed_ms": int, "tokens": int, "model": str},
        "degraded": bool,
        "refused": bool,   # 仅拒答时为 True
    }
    """
    chunks, prompt, degraded = _prepare(query, topk)
    if degraded:
        return degraded

    t0 = time.time()
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": MAX_TOKENS},
        )
    except Exception as e:
        return handle_llm_error(query, str(e))

    elapsed = int((time.time() - t0) * 1000)
    answer = resp["message"]["content"]
    tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)

    cited = extract_citations(answer, chunks)

    return {
        "answer": answer,
        "citations": cited,
        "metrics": {"elapsed_ms": elapsed, "tokens": tokens, "model": model},
        "degraded": False,
        "refused": False,
        "debug": validate_citations(answer, chunks),
    }


def rag_answer_stream(
    query: str,
    topk: int = 5,
    model: str = DEFAULT_MODEL,
) -> Iterator[str]:
    """流式 RAG 问答。逐 token 输出。"""
    chunks, prompt, degraded = _prepare(query, topk)
    if degraded:
        yield degraded["answer"]
        return

    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.1, "num_predict": MAX_TOKENS},
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except Exception as e:
        yield handle_llm_error(query, str(e))["answer"]


def rag_answer_with_citations(
    query: str,
    topk: int = 5,
    model: str = DEFAULT_MODEL,
) -> tuple:
    """
    流式 + 返回引用（用于前端：一边流式显示答案，一边准备引用列表）。
    """
    chunks, prompt, degraded = _prepare(query, topk)
    if degraded:
        return degraded, None

    collected = []

    def gen():
        try:
            stream = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": 0.1, "num_predict": MAX_TOKENS},
            )
            for chunk in stream:
                piece = chunk["message"]["content"]
                collected.append(piece)
                yield piece
        except Exception as e:
            yield handle_llm_error(query, str(e))["answer"]

    def finalize():
        answer = "".join(collected)
        return {
            "answer": answer,
            "citations": extract_citations(answer, chunks),
            "degraded": False,
        }

    return gen, finalize