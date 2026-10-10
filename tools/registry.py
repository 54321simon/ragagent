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
            return doc_id[: -len(ext)]
    return doc_id


def _available_doc_ids() -> list[str]:
    """返回当前知识库中所有 doc_id。"""
    try:
        return [d["doc_id"] for d in get_docs()]
    except Exception:
        return []


# ============ 1. 知识库 RAG 检索 ============
def rag_search(query: str, doc_ids: list = None) -> str:
    """在论文知识库中检索相关内容，返回带页码的片段。
    参数 query: 检索语句。问"方法"时请带上具体技术术语，如 'Transformer self-attention'。
    参数 doc_ids: 可选。限定在这些文档内检索，必须是 list_documents 返回的真实 doc_id。
                  None 或空列表 = 全部文档。
    适用场景：需要查找论文原文、事实性信息。当用户指定了一篇或多篇文档时，必须传 doc_ids。"""
    # ===== doc_id 存在性检查（硬性兜底）=====
    if doc_ids:
        normalized = [_normalize_doc_id(d) for d in doc_ids]
        available = _available_doc_ids()
        available_set = set(available)
        missing = [d for d in normalized if d not in available_set]
        if missing:
            return (
                f"错误：doc_id {missing} 不存在于知识库中。\n"
                f"知识库中实际的 doc_id 有：\n"
                + "\n".join(f"  - {d}" for d in available)
                + "\n\n请使用上述真实 doc_id 重新调用 rag_search。"
                f"如果用户没有指定具体文档，请先调用 list_documents 查看。"
            )
        doc_ids = normalized  # 用规范化后的版本继续

    from rag.chain import get_evidence

    chunks = get_evidence(query, topk=5, doc_ids=doc_ids)
    if not chunks:
        return "未检索到相关内容，请上传论文或缩小问题范围。"
    return "\n\n".join(
        f"{c.cite()} score={c.score:.4f}\n{c.text[:650]}" for c in chunks
    )


# ============ 2. 论文元信息提取 ============
_PAPER_META_CACHE = {}


def paper_meta(doc_id: str) -> str:
    """提取论文的标题、作者、年份、摘要、DOI。
    参数 doc_id: 文档ID，必须是 list_documents 返回的真实 doc_id（不要带 .pdf 后缀）。
    适用场景：只在用户明确问标题、作者、年份、DOI 时调用。"""
    from pathlib import Path
    import json
    from common.config import ROOT, PAPERS_DIR

    doc_id = _normalize_doc_id(doc_id)
    docs = get_docs()
    if not any(d["doc_id"] == doc_id for d in docs):
        return f"未找到文档 {doc_id}"
    manifest = ROOT / "experiments" / "corpus_manifest.json"
    records = (
        json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    )
    record = next((d for d in records if d.get("doc_id") == doc_id), {})
    source = record.get("first_page", "")
    if not source:
        import pymupdf

        path = PAPERS_DIR / next(d["doc_name"] for d in docs if d["doc_id"] == doc_id)
        if path.suffix.lower() == ".pdf" and path.exists():
            with pymupdf.open(path) as pdf:
                source = pdf[0].get_text()
                record["title"] = pdf.metadata.get("title") or "待人工核验"
                record["authors"] = (
                    pdf.metadata.get("author") or "首页未可靠提取，请核验"
                )
    abstract = re.search(
        r"abstract[.\s]*(.*?)(?:\n(?:1\s*\n)?introduction|\n1[ .]+Introduction)",
        source,
        re.I | re.S,
    )
    doi = re.search(r"10\.\d{4,9}/[^\s]+", source)
    arxiv = record.get("arxiv_id")
    return (
        f"【{doc_id}.pdf-第1页】\n"
        f"标题：{record.get('title', '待人工核验')}\n"
        f"作者：{record.get('authors', '首页未可靠提取，请核验')}\n"
        f"年份：{record.get('year', '首页未可靠提取，请核验')}\n"
        f"摘要：{abstract.group(1).strip()[:1300] if abstract else '请查看原文首页摘要'}\n"
        f"DOI：{doi.group() if doi else ('10.48550/arXiv.' + arxiv + '（arXiv记录DOI，非期刊DOI）' if arxiv else '原文首页未标注')}"
    )


# ============ 3. 论文对比（支持多篇） ============
def paper_compare(doc_ids: list) -> str:
    """对比多篇论文的方法、数据集、实验结果。
    参数 doc_ids: 文档ID列表，如 ['论文A', '论文B']，至少 2 篇。
                  必须是 list_documents 返回的真实 doc_id。
    适用场景：用户问 '论文A和B有什么不同'、'对比这几篇论文'。"""
    if not doc_ids or len(doc_ids) < 2:
        return (
            "对比需要至少 2 篇文档。请先用 list_documents 查看知识库中的文档，"
            "然后用真实 doc_id 调用（如 ['论文A', '论文B']）"
        )

    normalized = [_normalize_doc_id(d) for d in doc_ids]

    # doc_id 存在性检查
    available = _available_doc_ids()
    available_set = set(available)
    missing = [d for d in normalized if d not in available_set]
    if missing:
        return (
            f"错误：doc_id {missing} 不存在于知识库中。\n"
            f"知识库中实际的 doc_id 有：\n"
            + "\n".join(f"  - {d}" for d in available)
            + "\n\n请使用上述真实 doc_id 重新调用 paper_compare。"
        )

    from rag.chain import get_evidence

    results = []
    for doc_id in normalized:
        chunks = get_evidence(
            "method architecture dataset experiment results", topk=5, doc_ids=[doc_id]
        )
        text = "\n\n".join(f"{c.cite()} {c.text[:600]}" for c in chunks)
        results.append(f"论文 {doc_id}\n{text or '未检索到内容'}")
    return (
        "\n\n".join(results)
        + "\n请按方法、数据集、实验条件和结果对比，不能把不同训练设置的数字直接视为公平比较。"
    )


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
    参数 doc_id: 文档ID，必须是 list_documents 返回的真实 doc_id（不要带 .pdf 后缀）。
    适用场景：用户问 '这篇论文讲了什么'、'总结一下这篇论文'。"""
    doc_id = _normalize_doc_id(doc_id)
    docs = get_docs()
    if not any(d["doc_id"] == doc_id for d in docs):
        return f"未找到文档 {doc_id}，可用文档：{[d['doc_id'] for d in docs]}"

    from rag.chain import get_evidence

    chunks = get_evidence(
        "introduction method architecture experiment results conclusion",
        topk=6,
        doc_ids=[doc_id],
    )
    return "请根据证据整理背景、方法、结果和结论，并逐项保留引用。\n" + "\n\n".join(
        f"{c.cite()} {c.text[:650]}" for c in chunks
    )


def _split_sections(text: str, doc_id: str) -> str:
    """从全文里粗略切出背景/方法/结果/结论。"""

    def extract(pattern: str, max_len: int = 400) -> str:
        m = re.search(pattern, text, re.I)
        if not m:
            return "未找到"
        return text[m.start() : m.start() + max_len].replace("\n", " ")

    background = extract(r"(?:introduction|background)")
    method = extract(r"(?:method|approach|model architecture)")
    result = extract(r"(?:experiment|evaluation|result)")
    conclusion = extract(r"(?:conclusion|discussion|future work)")

    return (
        f"【背景】{background}\n\n"
        f"【方法】{method}\n\n"
        f"【结果】{result}\n\n"
        f"【结论】{conclusion}"
    )


# ============ 6. 当前时间 ============
def current_time() -> str:
    """返回当前系统时间。
    适用场景：用户问 '今天几号'、'现在几点'，或需要时间戳。"""
    from datetime import timezone, timedelta

    return datetime.now(timezone(timedelta(hours=8))).strftime(
        "%Y-%m-%d %H:%M:%S UTC+08:00"
    )


# ============ 7. 计算器 ============
def calculator(expr: str) -> str:
    """执行简单数学运算。支持 + - * / ** () 等。
    参数 expr: 数学表达式，如 '3.14*2.56'、'(1+2)**3'。
    适用场景：对比实验数据、计算指标。"""
    import ast, operator, math

    if len(expr) > 120:
        return "计算错误：表达式过长"
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = evaluate(node.operand) * (
                -1 if isinstance(node.op, ast.USub) else 1
            )
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            a, b = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 16:
                raise ValueError("指数超过允许范围")
            value = binary[type(node.op)](a, b)
        else:
            raise ValueError("只支持数值和四则运算、幂、取余")
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("结果超过允许范围")
        return value

    try:
        tree = ast.parse(expr, mode="eval")
        if len(list(ast.walk(tree))) > 64:
            raise ValueError("表达式过于复杂")
        result = evaluate(tree.body)
        display = format(result, ".12g") if isinstance(result, float) else str(result)
        return f"{expr} = {display}"
    except Exception as e:
        return f"计算错误：{e}"


# ============ 8. 文档列表 ============
def list_documents() -> str:
    """列出知识库中所有已上传的文档，返回真实的 doc_id。
    适用场景：用户问 '有哪些论文'、'知识库里有什么'，
             或者用户提到'这两篇论文''这些论文'但未指明具体是哪几篇时，必须先调用此工具。"""
    docs = get_docs()
    if not docs:
        return "知识库为空，请先上传论文。"
    return (
        "已上传文档（调用 rag_search / paper_compare / summarize_paper 时，"
        "doc_id 必须严格用下面的值）：\n" + "\n".join(f"- {d['doc_id']}" for d in docs)
    )


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
            ptype = (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            )
            default = (
                f"={param.default!r}"
                if param.default != inspect.Parameter.empty
                else ""
            )
            params.append(f"{pname}: {ptype}{default}")
        param_str = ", ".join(params) if params else ""
        doc = (fn.__doc__ or "无描述").strip()
        first_line = doc.split("\n")[0]
        lines.append(f"- {name}({param_str}): {first_line}")
    return "\n".join(lines)
