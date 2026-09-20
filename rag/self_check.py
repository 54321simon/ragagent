"""RAG 自检：判断检索到的 chunk 能否回答用户问题。
用 qwen2.5:7b 做一次轻量 LLM 判断，避免库里没答案时硬凑。
"""
import ollama

MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

_PROMPT = """判断下面「参考片段」能否回答「用户问题」。

规则：
1. 只有片段里明确包含回答问题的具体信息时，才回答"能"。
2. 片段只是提到了相关概念但没有具体答案 → 回答"不能"。
3. 片段和问题完全无关 → 回答"不能"。
4. 只回答"能"或"不能"，不要解释。

用户问题：{query}

参考片段：
{context}

回答："""


def check_answerable(query: str, chunks: list, max_chunks: int = 3) -> bool:
    """True = 能回答；False = 应拒答。"""
    if not chunks:
        return False

    lines = []
    for c in chunks[:max_chunks]:
        text = c.text.strip().replace("\n", " ")
        lines.append(f"[{c.cite()}] {text[:300]}")
    context = "\n".join(lines)

    prompt = _PROMPT.format(query=query, context=context)

    try:
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        answer = resp["message"]["content"].strip()
        return ("能" in answer) and ("不能" not in answer)
    except Exception:
        # LLM 失败时保守放行，让下游兜底
        return True