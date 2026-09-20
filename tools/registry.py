"""Agent 可用工具集。共 8 个工具，统一注册。
工具函数必须有 docstring（Agent 靠它理解工具用途）。
"""
import time
import re
import inspect
from datetime import datetime
from typing import Any

from retriever.api import retrieve_best, get_docs
from rag.chain import rag_answer


# ============ 工具函数 ============

def _normalize_doc_id(doc_id: str) -> str:
    """模型经常传 'sample.pdf'，但真实 doc_id 是 'sample'。统一处理。"""
    for ext in [".pdf", ".docx", ".txt", ".md", ".doc"]:
        if doc_id.endswith(ext):
            return doc_id[:-len(ext)]
    return doc_id


# ============ 1. 知识库 RAG 检索 ============
def rag_search(query: str, doc_ids: list = None) -> str:
    """在论文知识库中检索相关内容，返回带页码的片段。
    参数 query: 检索语句。问"方法"时请带上具体技术术语，如 'Transformer self-attention'。
    参数 doc_ids: 可选。限定在这些文档内检索，如 ['sample', 'sample2']。
                  None 或空列表 = 全部文档。
    适用场景：需要查找论文原文、事实性信息。当用户指定了一篇或多篇文档时，必须传 doc_ids。"""
    # 根据问题类型自动扩展 query
    expanded = query

    if any(w in query for w in ["方法", "method", "approach", "怎么做的", "如何实现"]):
        expanded = f"{query} Transformer architecture encoder decoder self-attention"
    elif any(w in query for w in ["结果", "result", "实验", "experiment", "性能"]):
        expanded = f"{query} BLEU WMT experiment dataset results"
    elif any(w in query for w in ["结论", "conclusion", "总结", "未来"]):
        expanded = f"{query} conclusion future work"

    if any(w in query for w in ["整体架构", "总体架构", "模型结构", "模型架构"]):
        expanded = "Transformer model architecture encoder decoder Figure 1"

    if any(w in query for w in ["注意力机制", "attention mechanism", "用了哪些注意力"]):
        expanded = ("multi-head attention scaled dot-product attention "
                    "encoder-decoder self-attention")

    if any(w in query for w in ["未来", "future", "展望", "下一步"]):
        expanded = "conclusion future work applications"

    chunks = retrieve_best(expanded, topk=5)

    # doc_ids 过滤：如果指定了文档列表，只保留这些文档的 chunk
    if doc_ids:
        doc_id_set = {_normalize_doc_id(d) for d in doc_ids}
        filtered = [c for c in chunks if c.doc_id in doc_id_set]
        if not filtered:
            # 这些文档的 chunk 没进 top-5，扩大召回范围再过滤
            all_chunks = retrieve_best(expanded, topk=50)
            filtered = [c for c in all_chunks if c.doc_id in doc_id_set][:5]
        if not filtered:
            return f"未在文档 {list(doc_id_set)} 中找到与「{query}」相关的内容"
        chunks = filtered

    # 去重
    seen = set()
    unique = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)

    if not unique:
        return "未检索到相关内容"

    lines = []
    for c in unique[:3]:
        text = c.text.strip().replace("\n", " ")
        lines.append(f"{c.cite()} {text[:250]}")
    return "\n".join(lines)


# ============ 2. 论文元信息提取 ============
_PAPER_META_CACHE = {}


def paper_meta(doc_id: str) -> str:
    """提取论文的标题、作者、年份、摘要、DOI。
    参数 doc_id: 文档ID，如 'sample'（不要带 .pdf 后缀）。
    适用场景：只在用户明确问标题、作者、年份、DOI 时调用。"""
    doc_id = _normalize_doc_id(doc_id)
    if doc_id in _PAPER_META_CACHE:
        return _PAPER_META_CACHE[doc_id]

    docs = get_docs()
    if not any(d["doc_id"] == doc_id for d in docs):
        return f"未找到文档 {doc_id}，可用文档：{[d['doc_id'] for d in docs]}"

    from retriever.store import get_collection
    col = get_collection()
    data = col.get(
        where={"$and": [{"doc_id": doc_id}, {"page": 1}]},
        include=["documents"],
    )

    if not data["ids"]:
        return f"文档 {doc_id} 没有第 1 页内容"

    first_page_text = "\n".join(data["documents"])
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    title = "未知"
    for l in lines[:15]:
        if 10 <= len(l) <= 150 and "@" not in l and "http" not in l and "arXiv" not in l:
            title = l
            break

    year_match = re.search(r"(201[5-9]|202[0-9]|2030)", first_page_text)
    year = year_match.group() if year_match else "未知"

    abstract = "未找到"
    abs_match = re.search(r"Abstract\s*(.{50,300})", first_page_text, re.S | re.I)
    if abs_match:
        abstract = abs_match.group(1).strip().replace("\n", " ")[:300]

    result = (f"**文档**: `{doc_id}`\n\n"
              f"**标题（推测）**: {title}\n\n"
              f"**年份（推测）**: {year}\n\n"
              f"**作者**: 需从 PDF 元数据提取（当前未实现）\n\n"
              f"**摘要（前 300 字）**: {abstract}\n\n"
              f"**DOI**: 需从 PDF 元数据提取（当前未实现）")

    _PAPER_META_CACHE[doc_id] = result
    return result


# ============ 3. 论文对比（支持多篇） ============
def paper_compare(doc_ids: list) -> str:
    """对比多篇论文的方法、数据集、实验结果。
    参数 doc_ids: 文档ID列表，如 ['sample', 'sample2']，至少 2 篇。
    适用场景：用户问 '论文A和B有什么不同'、'对比这几篇论文'。"""
    if not doc_ids or len(doc_ids) < 2:
        return "对比需要至少 2 篇文档，请指定 doc_ids（如 ['sample', 'sample2']）"

    normalized = [_normalize_doc_id(d) for d in doc_ids]

    results = []
    for doc_id in normalized:
        chunks = retrieve_best(f"{doc_id} method dataset experiment", topk=10)
        doc_chunks = [c for c in chunks if c.doc_id == doc_id][:3]
        text = "\n".join(c.text[:200] for c in doc_chunks)
        results.append(f"【{doc_id}】\n{text or '未检索到内容'}")

    return ("\n\n".join(results)
            + "\n\n提示：请基于以上片段，从方法、数据集、实验结果三个维度对比。")


# ============ 4. 关键词提取 ============
def extract_keywords(text: str, topk: int = 5) -> str:
    """从文本中提取核心关键词。
    参数 text: 待提取的文本；topk: 返回前几个关键词，默认 5。
    适用场景：用户问 '这篇论文的关键词是什么'，或者需要理解用户问题的核心概念。"""
    import jieba.analyse
    keywords = jieba.analyse.extract_tags(text, topK=topk)
    return f"关键词: {', '.join(keywords)}"


# ============ 5. 摘要生成 ============
def summarize_paper(doc_id: str) -> str:
    """针对特定论文生成结构化摘要：背景-方法-结果-结论。
    参数 doc_id: 文档ID，如 'sample'（不要带 .pdf 后缀）。
    适用场景：用户问 '这篇论文讲了什么'、'总结一下这篇论文'。"""
    doc_id = _normalize_doc_id(doc_id)
    docs = get_docs()
    if not any(d["doc_id"] == doc_id for d in docs):
        return f"未找到文档 {doc_id}，可用文档：{[d['doc_id'] for d in docs]}"

    from retriever.store import get_collection
    col = get_collection()
    data = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])

    if not data["ids"]:
        return f"文档 {doc_id} 没有任何 chunk"

    full_text = "\n".join(data["documents"])
    return _split_sections(full_text, doc_id)


def _split_sections(text: str, doc_id: str) -> str:
    """从全文里粗略切出背景/方法/结果/结论。"""
    def extract(pattern: str, max_len: int = 400) -> str:
        m = re.search(pattern, text, re.I)
        if not m:
            return "未找到"
        return text[m.start():m.start() + max_len].replace("\n", " ")

    background = extract(r"(?:introduction|background)")
    method = extract(r"(?:method|approach|model architecture)")
    result = extract(r"(?:experiment|evaluation|result)")
    conclusion = extract(r"(?:conclusion|discussion|future work)")

    return (f"【背景】{background}\n\n"
            f"【方法】{method}\n\n"
            f"【结果】{result}\n\n"
            f"【结论】{conclusion}")


# ============ 6. 当前时间 ============
def current_time() -> str:
    """返回当前系统时间。
    适用场景：用户问 '今天几号'、'现在几点'，或需要时间戳。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============ 7. 计算器 ============
def calculator(expr: str) -> str:
    """执行简单数学运算。支持 + - * / ** () 等。
    参数 expr: 数学表达式，如 '3.14*2.56'、'(1+2)**3'。
    适用场景：对比实验数据、计算指标。"""
    allowed = {"__builtins__": {}}
    try:
        if not re.match(r"^[\d\s\+\-\*/\(\)\.\*\*]+$", expr):
            return f"表达式包含非法字符：{expr}"
        result = eval(expr, allowed, {})
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# ============ 8. 文档列表 ============
def list_documents() -> str:
    """列出知识库中所有已上传的文档。
    适用场景：用户问 '有哪些论文'、'知识库里有什么'。"""
    docs = get_docs()
    if not docs:
        return "知识库为空，请先上传论文。"
    return "已上传文档：\n" + "\n".join(f"- {d['doc_id']} ({d['doc_name']})" for d in docs)


# ============ 工具注册表 ============
def get_all_tools() -> dict:
    """返回所有工具的字典 {tool_name: function}。"""
    return {
        "rag_search": rag_search,
        "paper_meta": paper_meta,
        "paper_compare": paper_compare,
        "extract_keywords": extract_keywords,
        "summarize_paper": summarize_paper,
        "current_time": current_time,
        "calculator": calculator,
        "list_documents": list_documents,
    }


def get_tool_descriptions() -> str:
    """生成工具描述文本，给 Agent 的 System Prompt 用。"""
    tools = get_all_tools()
    lines = []
    for name, fn in tools.items():
        sig = inspect.signature(fn)
        params = []
        for pname, param in sig.parameters.items():
            ptype = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"
            default = f"={param.default!r}" if param.default != inspect.Parameter.empty else ""
            params.append(f"{pname}: {ptype}{default}")
        param_str = ", ".join(params) if params else ""
        doc = (fn.__doc__ or "无描述").strip()
        first_line = doc.split("\n")[0]
        lines.append(f"- {name}({param_str}): {first_line}")
    return "\n".join(lines)