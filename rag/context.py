"""上下文智能拼接。控制 token 长度，保留高相关性片段。"""
from typing import List
from common.schemas import RetrievedChunk
from rag.prompts import CHUNK_TEMPLATE

# 粗略估算：1 个中文字符 ≈ 1 token，1 个英文字符 ≈ 0.25 token
# 保守估计：1 字符 ≈ 0.6 token
MAX_CONTEXT_CHARS = 3000   # 约 1800 token，给回答留空间


def build_context(chunks: List[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    按相关性排序拼接上下文，超长自动截断。
    chunks 应已按 score 降序排列。
    """
    if not chunks:
        return ""

    parts = []
    total = 0
    for c in chunks:
        piece = CHUNK_TEMPLATE.format(
            doc_name=c.doc_name,
            page=c.page,
            text=c.text.strip(),
        )
        if total + len(piece) > max_chars:
            # 截断最后一个片段
            remain = max_chars - total
            if remain > 100:  # 至少保留 100 字
                parts.append(piece[:remain] + "...[截断]")
            break
        parts.append(piece)
        total += len(piece)

    return "\n".join(parts)