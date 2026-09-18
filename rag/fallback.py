"""RAG 降级策略。模拟真实系统的鲁棒性。"""
from typing import List
from common.schemas import RetrievedChunk


def handle_no_results(query: str) -> dict:
    """降级 1：检索完全无结果。"""
    return {
        "answer": "根据当前知识库未找到相关文档，无法回答该问题。\n\n"
                  "建议：\n"
                  "1. 检查是否已上传相关论文\n"
                  "2. 换一种问法\n"
                  "3. 如果问题超出知识库范围，请尝试联网搜索",
        "citations": [],
        "degraded": True,
        "degrade_type": "no_results",
    }


def handle_low_relevance(query: str, chunks: List[RetrievedChunk],
                         threshold: float = 0.3) -> dict:
    """降级 2：检索结果相关性低。"""
    best = max((c.score for c in chunks), default=0.0)
    return {
        "answer": f"未找到高相关性内容（最高相似度 {best:.3f}，低于阈值 {threshold}）。\n\n"
                  f"以下是可能相关的片段，供您参考确认：",
        "citations": chunks,
        "degraded": True,
        "degrade_type": "low_relevance",
    }


def handle_llm_error(query: str, error: str) -> dict:
    """降级 3：LLM 调用失败。"""
    return {
        "answer": f"抱歉，生成回答时出现错误：{error}\n\n"
                  f"建议：\n"
                  f"1. 检查 Ollama 是否运行（`ollama ps`）\n"
                  f"2. 确认模型已下载（`ollama list`）\n"
                  f"3. 稍后重试",
        "citations": [],
        "degraded": True,
        "degrade_type": "llm_error",
    }