"""系统健康检查。检查 Ollama / 模型 / Chroma / 检索链路状态。"""
import time
import ollama


def check_ollama() -> dict:
    """检查 Ollama 服务是否运行。"""
    try:
        t0 = time.time()
        models = ollama.list()
        elapsed = int((time.time() - t0) * 1000)
        model_names = [m.get("model") or m.get("name") for m in models.get("models", [])]
        return {
            "ok": True,
            "message": f"Ollama 运行中（{len(model_names)} 个模型）",
            "models": model_names,
            "elapsed_ms": elapsed,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": f"Ollama 未运行：{e}",
            "models": [],
            "elapsed_ms": 0,
        }


def check_model(model_name: str) -> dict:
    """检查指定模型是否已拉取。"""
    try:
        models = ollama.list()
        model_names = [m.get("model") or m.get("name") for m in models.get("models", [])]
        found = any(model_name in m for m in model_names)
        return {
            "ok": found,
            "message": f"模型 {model_name} {'已就绪' if found else '未找到'}",
        }
    except Exception as e:
        return {"ok": False, "message": f"检查失败：{e}"}


def check_chroma() -> dict:
    """检查 Chroma 向量库。"""
    try:
        from retriever.store import get_collection, list_docs
        col = get_collection()
        count = col.count()
        docs = list_docs()
        return {
            "ok": True,
            "message": f"向量库正常（{count} chunks / {len(docs)} 文档）",
            "count": count,
            "docs": [d["doc_id"] for d in docs],
        }
    except Exception as e:
        return {"ok": False, "message": f"向量库异常：{e}", "count": 0, "docs": []}


def check_retrieval() -> dict:
    """检查检索链路（跑一次最小检索）。"""
    try:
        from retriever.api import retrieve_best
        t0 = time.time()
        results = retrieve_best("测试", topk=1)
        elapsed = int((time.time() - t0) * 1000)
        return {
            "ok": True,
            "message": f"检索链路正常（{elapsed}ms）",
            "elapsed_ms": elapsed,
            "hits": len(results),
        }
    except Exception as e:
        return {"ok": False, "message": f"检索链路异常：{e}", "elapsed_ms": 0, "hits": 0}


def run_all_checks(model_name: str = "qwen2.5:7b-instruct-q4_K_M",
                   embedding_model: str = "bge-m3:567m") -> list[dict]:
    """运行所有检查，返回结果列表。"""
    checks = [
        {"name": "Ollama 服务", **check_ollama()},
        {"name": f"LLM 模型", **check_model(model_name)},
        {"name": f"Embedding 模型", **check_model(embedding_model)},
        {"name": "Chroma 向量库", **check_chroma()},
        {"name": "检索链路", **check_retrieval()},
    ]
    return checks