import json, time, threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import pytest
from agent import react_loop as agent
from rag.cache import SemanticCache
from common.schemas import RetrievedChunk
from tools.registry import calculator
from experiments.evaluate import score


def test_parallel_real_overlap():
    barrier = threading.Barrier(2)

    def tool(value):
        barrier.wait(timeout=1)
        return str(value)

    res = agent._execute_tools_parallel(
        [{"action": "t", "action_input": {"value": v}} for v in (1, 2)], {"t": tool}
    )
    assert [r["observation"] for r in res] == ["1", "2"] and all(r["ok"] for r in res)


def test_timeout_returns_without_wait():
    t = time.perf_counter()
    res = agent._execute_tool(
        "slow", {}, {"slow": lambda: time.sleep(0.25)}, timeout_sec=0.02
    )
    assert not res[2] and time.perf_counter() - t < 0.15


def test_argument_error_and_unknown_tool():
    assert not agent._execute_tool("x", {"bad": 1}, {"x": lambda good: good})[2]
    assert not agent._execute_tool("missing", {}, {})[2]


def test_transient_retry():
    count = []

    def tool():
        count.append(1)
        if len(count) == 1:
            raise ConnectionError("temporary")
        return "recovered"

    assert agent._execute_tool("tool", {}, {"tool": tool})[2] and len(count) == 2


@pytest.mark.parametrize(
    "expr,answer",
    [("3.14*2.56", "8.0384"), ("(1+2)**3", "27"), ("-2+4", "2"), ("10%3", "1")],
)
def test_calculator(expr, answer):
    assert answer in calculator(expr)


@pytest.mark.parametrize("expr", ['__import__("os")', "9**999999", "1/0", "[1]*99999"])
def test_calculator_rejects(expr):
    assert calculator(expr).startswith("计算错误")


def test_doc_page_identity():
    assert agent._extract_pages("【A.pdf-第2页】【B.pdf-第2页】") == {
        ("A.pdf", 2),
        ("B.pdf", 2),
    }


def test_multi_document_metrics():
    retrieved = [
        RetrievedChunk("a", "A", "A.pdf", 2, "x"),
        RetrievedChunk("b", "B", "B.pdf", 2, "y"),
    ]
    gold = [{"doc_id": "A", "page": 2}, {"doc_id": "B", "page": 2}]
    assert score(retrieved, gold)["recall@5"] == 1
    assert score(retrieved[:1], gold)["recall@5"] == 0.5
    assert score([retrieved[0]], [{"doc_id": "B", "page": 2}])["hit@5"] == 0


def test_cache_scope_semantic_and_numbers(tmp_path):
    c = SemanticCache(tmp_path / "cache.db", threshold=0.95)
    c.put("请介绍这种方法", "scope-a", [1, 0], {"answer": "ok"})
    assert c.get("介绍一下这种方法", "scope-a", [0.999, 0.001])[0]["answer"] == "ok"
    assert c.get("介绍一下这种方法", "scope-b", [1, 0])[0] is None
    c.put("MAE masking 75%", "numbers", [1, 0], {"answer": "75"})
    assert c.get("MAE masking 50%", "numbers", [1, 0])[0] is None
    c.clear()
    assert c.get("请介绍这种方法", "scope-a")[0] is None


def test_cache_expiration(tmp_path):
    c = SemanticCache(tmp_path / "ttl.db", ttl=0)
    c.put("q", "scope", [1], {"answer": "ok"})
    assert c.get("q", "scope")[0] is None


def test_history_hard_budget():
    with patch.object(agent, "summarize_history", return_value="摘要" * 10):
        original = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "测" * 4000}
            for i in range(20)
        ]
        history, summary = agent.trim_history(original)
        assert (
            sum(len(h["content"]) for h in history) + len(summary)
            <= agent.MAX_HISTORY_TOKENS
        )
        assert len(original[0]["content"]) == 4000


@pytest.mark.parametrize("entry", ["sync", "stream"])
def test_shared_agent_parallel_and_tokens(entry, monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "0")
    barrier = threading.Barrier(2)

    def tool(doc):
        barrier.wait(timeout=1)
        return f"【{doc}.pdf-第2页】 evidence"

    responses = iter(
        [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "thought": "parallel",
                            "actions": [
                                {"action": "tool", "action_input": {"doc": "A"}},
                                {"action": "tool", "action_input": {"doc": "B"}},
                            ],
                        }
                    )
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            {
                "message": {"content": json.dumps({"final_answer": "ready"})},
                "prompt_eval_count": 20,
                "eval_count": 5,
            },
        ]
    )

    def chat(**kw):
        if kw.get("stream"):
            return iter(
                [
                    {
                        "message": {
                            "content": "依据A【A.pdf-第2页】和B【B.pdf-第2页】"
                        },
                        "done": True,
                        "prompt_eval_count": 30,
                        "eval_count": 10,
                    }
                ]
            )
        return next(responses)

    with patch.object(agent.ollama, "chat", side_effect=chat):
        if entry == "sync":
            res = agent.react_loop("compare", tools={"tool": tool})
        else:
            res = list(agent.react_loop_stream("compare", tools={"tool": tool}))[-1]
    assert len([s for s in res["trace"] if s.action == "tool"]) == 2
    assert (
        res["metrics"]["total_tokens"] == 80
        and res["metrics"]["invalid_citations"] == 0
    )
    assert (
        sum(v.get("tokens", 0) for k, v in res["tool_stats"].items() if k != "_meta")
        == 15
    )


def test_cache_invalidation_after_corpus_change(tmp_path):
    c = SemanticCache(tmp_path / "v.db")
    c.put("q", "corpus-v1", [1], {"answer": "old"})
    assert c.get("q", "corpus-v2", [1])[0] is None


def test_scoped_retrieval_before_ranking(monkeypatch):
    import retriever.api as api

    calls = []

    def query(q, topk, doc_ids):
        calls.append(doc_ids)
        return []

    monkeypatch.setattr(api, "query_chunks", query)
    assert api.retrieve("q", 5, ["B"]) == [] and calls == [["B"]]


def test_ingestion_retry(monkeypatch):
    import retriever.api as api

    calls = []

    def ingest(*a, **k):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("temporary")
        return 3

    monkeypatch.setattr(api, "ingest_pdf", ingest)
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    assert api.ingest_with_retry("x") == 3 and len(calls) == 2


def test_hallucinated_citation_is_detected():
    from rag.citation import validate_citations

    chunks = [
        RetrievedChunk(
            chunk_id="a",
            doc_id="MAE",
            doc_name="MAE.pdf",
            page=1,
            text="evidence",
            score=1,
        )
    ]
    checked = validate_citations("【MAE.pdf-第99页】", chunks)
    assert checked["hallucinated"] == [("MAE.pdf", 99)]
    assert checked["citation_count"] == 0


def test_citation_validation_is_serializable():
    from rag.citation import validate_citations

    chunks = [
        RetrievedChunk(
            chunk_id="a",
            doc_id="MAE",
            doc_name="MAE.pdf",
            page=1,
            text="evidence",
            score=1,
        )
    ]
    assert "MAE.pdf" in json.dumps(validate_citations("【MAE.pdf-第1页】", chunks))


def test_document_routes_preserve_independent_scopes():
    route = agent.document_route("归纳共同方法", ["MAE", "Swin"])
    assert [a["action_input"]["doc_ids"] for a in route["actions"]] == [
        ["MAE"],
        ["Swin"],
    ]
    assert (
        agent.document_route("比较方法区别", ["MAE", "Swin"])["action"]
        == "paper_compare"
    )
    assert agent.document_route("作者和年份", ["MAE"])["action"] == "paper_meta"


def test_private_information_does_not_route_to_metadata():
    route = agent.document_route("作者的私人手机号是什么", ["MAE"])
    assert route["action"] == "rag_search"


def test_repeated_question_cache_context_and_followup():
    q = "MAE 的作者是什么"
    history = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": "verified answer"},
    ]
    assert agent.cache_history(history, q) == []
    assert len(history) == 2
    assert agent.cache_history(history, "它的作者是什么") == history
