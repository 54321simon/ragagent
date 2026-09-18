"""引用处理。从答案中提取引用，校验是否来自检索片段。"""
import re
from typing import List
from common.schemas import RetrievedChunk

# 匹配【文档名-第X页】
CITE_PATTERN = re.compile(r"【([^】]+?)-第(\d+)页】")


def extract_citations(answer: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    从答案中提取引用，返回去重后的 RetrievedChunk 列表。
    只保留答案中实际出现的引用。
    """
    cited = set()
    for m in CITE_PATTERN.finditer(answer):
        doc_name, page = m.group(1), int(m.group(2))
        cited.add((doc_name, page))

    result = []
    seen = set()
    for c in chunks:
        key = (c.doc_name, c.page)
        if key in cited and key not in seen:
            result.append(c)
            seen.add(key)
    return result


def validate_citations(answer: str, chunks: List[RetrievedChunk]) -> dict:
    """
    校验引用：
    - 提取的引用
    - 幻觉引用（答案中标注了但检索片段里没有的）
    - 未引用片段（检索到但没用上的）
    """
    cited = extract_citations(answer, chunks)
    cited_keys = {(c.doc_name, c.page) for c in cited}
    all_keys = {(c.doc_name, c.page) for c in chunks}

    return {
        "cited": cited,
        "hallucinated": list(cited_keys - all_keys),   # 幻觉引用
        "unused": list(all_keys - cited_keys),         # 未使用片段
        "citation_count": len(cited_keys),
    }