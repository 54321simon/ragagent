"""Agent 可用工具集。共 8 个工具，统一注册。
工具函数必须有 docstring（Agent 靠它理解工具用途）。
"""
import time
import re
import inspect
from datetime import datetime
from typing import Any

from retriever.api import retrieve_hybrid, get_docs
from rag.chain import rag_answer


# ============ 工具函数 ============

def _normalize_doc_id(doc_id: str) -> str:
    """模型经常传 'sample.pdf'，但真实 doc_id 是 'sample'。统一处理。"""
    for ext in [".pdf", ".docx", ".txt", ".md"]:
        if doc_id.endswith(ext):
            return doc_id[:-len(ext)]
    return doc_id


# ============ 1. 知识库 RAG 检索 ============
def rag_search(query: str) -> str:
    """在论文知识库中检索相关内容，返回带页码的片段。
    参数 query: 检索语句。问"方法"时请带上具体技术术语，如 'Transformer self-attention'。
    适用场景：需要查找论文原文、事实性信息。"""
    # 根据问题类型自动扩展 query
    expanded = query
    if any(w in query for w in ["方法", "method", "approach", "怎么做的", "如何实现"]):
        expanded = f"{query} Transformer architecture encoder decoder self-attention"
    elif any(w in query for w in ["结果", "result", "实验", "experiment", "性能"]):
        expanded = f"{query} BLEU WMT experiment dataset results"
    elif any(w in query for w in ["结论", "conclusion", "总结", "未来"]):
        expanded = f"{query} conclusion future work"

    chunks = retrieve_hybrid(expanded, topk=5)

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
def paper_meta(doc_id: str) -> str:
    """提取论文的标题、作者、年份、摘要、DOI。
    参数 doc_id: 文档ID，如 'sample'（不要带 .pdf 后缀）。
    适用场景：需要论文的基本信息。"""
    doc_id = _normalize_doc_id(doc_id)
    docs = get_docs()
    if not any(d["doc_id"] == doc_id for d in docs):
        return f"未找到文档 {doc_id}，可用文档：{[d['doc_id'] for d in docs]}"

    chunks = retrieve_hybrid(doc_id, topk=1)
    first_text = chunks[0].text if chunks else ""

    lines = [l.strip() for l in first_text.split("\n") if l.strip()]
    title = lines[0] if lines else "未知"
    year_match = re.search(r"(19|20)\d{2}", first_text)
    year = year_match.group() if year_match else "未知"

    return (f"文档: {doc_id}\n"
            f"标题（推测）: {title[:100]}\n"
            f"年份（推测）: {year}\n"
            f"作者: [需从 PDF 元数据提取]\n"
            f"摘要: [需 PDF 摘要区块识别]\n"
            f"DOI: [需 PDF 元数据提取]")


# ============ 3. 论文对比 ============
def paper_compare(doc_id_a: str, doc_id_b: str) -> str:
    """对比两篇论文的方法、数据集、实验结果。
    参数 doc_id_a: 第一篇文档ID；doc_id_b: 第二篇文档ID。
    适用场景：用户问 '论文A和B有什么不同'。"""
    doc_id_a = _normalize_doc_id(doc_id_a)
    doc_id_b = _normalize_doc_id(doc_id_b)

    a_chunks = retrieve_hybrid(f"{doc_id_a} method dataset experiment", topk=3)
    b_chunks = retrieve_hybrid(f"{doc_id_b} method dataset experiment", topk=3)

    a_text = "\n".join(c.text[:200] for c in a_chunks if c.doc_id == doc_id_a)
    b_text = "\n".join(c.text[:200] for c in b_chunks if c.doc_id == doc_id_b)

    return (f"【{doc_id_a}】\n{a_text or '未检索到内容'}\n\n"
            f"【{doc_id_b}】\n{b_text or '未检索到内容'}\n\n"
            f"提示：请基于以上片段，从方法、数据集、实验结果三个维度对比。")


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

    # 直接拉该文档的所有 chunks（避免被其他文档干扰）
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
    """生成工具描述文本，给 Agent 的 System Prompt 用。
    包含：工具名、功能描述、参数名及类型。"""
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