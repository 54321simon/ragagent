"""手写 ReAct 循环。禁止使用 LangChain Agent 高层封装。

核心流程：
    Thought → Action → Observation → 循环 → Final Answer
支持多轮对话（history 参数）和文档范围限定（doc_ids 参数）。
内置两层硬性保护：
  1. rag_search 本轮无新增页码 → 强制 final_answer
  2. rag_search 累计召回页数 >= MAX_TOTAL_PAGES → 强制 final_answer

本文件同时提供：
  - react_loop          ：同步阻塞版（评测脚本 / demo_agent.py 使用）
  - react_loop_stream   ：流式事件版（Streamlit UI 使用，思路一）

日志：
  - 每次对话结束后写入 data/logs/agent_YYYYMMDD.jsonl
  - 可通过环境变量 LOG_DIR 覆盖目录，LOG_ENABLED=0 关闭
"""
import json
import os
import re
import time
import inspect
from datetime import datetime
from typing import Callable, Iterator
import ollama

from common.schemas import AgentStep
from tools.registry import get_all_tools, get_tool_descriptions

MAX_ITER = 4
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
MAX_TOTAL_PAGES = 5

# RAG 命中率阈值：至少一个 chunk 的 score 超过这个值才算“命中”
RAG_HIT_SCORE_THRESHOLD = 0.5


# ============ 日志 ============
_LOG_DIR = os.environ.get(
    "LOG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "logs"),
)
_LOG_ENABLED = os.environ.get("LOG_ENABLED", "1") != "0"


def _step_to_dict(step) -> dict:
    """把 AgentStep 转成可 JSON 序列化的 dict。"""
    try:
        return {
            "step_idx": getattr(step, "step_idx", None),
            "thought": getattr(step, "thought", ""),
            "action": getattr(step, "action", ""),
            "action_input": getattr(step, "action_input", {}),
            "observation": getattr(step, "observation", ""),
            "elapsed_ms": getattr(step, "elapsed_ms", 0),
        }
    except Exception:
        return {}


def _write_log(record: dict):
    """把一条记录追加到按天分文件的 JSONL。失败不影响主流程。"""
    if not _LOG_ENABLED:
        return
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fname = f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"
        path = os.path.join(_LOG_DIR, fname)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 日志失败不能影响主流程
        pass


def _build_log_record(
    question: str,
    answer: str,
    trace: list,
    metrics: dict,
    tool_stats: dict,
    history_append: list,
    success: bool,
    doc_ids: list | None,
    session_id: str | None,
    model: str,
    extra: dict | None = None,
) -> dict:
    """组装日志记录。"""
    # 从 answer 里提取引用
    citations = []
    for m in re.finditer(r"【([^】]+?)-第(\d+)页】", answer or ""):
        citations.append({
            "doc_name": m.group(1),
            "page": int(m.group(2)),
        })
    # 去重
    seen = set()
    uniq = []
    for c in citations:
        key = (c["doc_name"], c["page"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id or "anonymous",
        "question": question,
        "doc_ids": doc_ids,
        "answer": answer,
        "citations": uniq,
        "trace": [_step_to_dict(s) for s in (trace or [])],
        "metrics": metrics,
        "tool_stats": _sanitize_tool_stats(tool_stats),
        "success": success,
        "model": model,
    }
    if extra:
        rec.update(extra)
    return rec


def _sanitize_tool_stats(stats: dict) -> dict:
    """确保 tool_stats 可 JSON 序列化。"""
    if not stats:
        return {}
    out = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()
                      if isinstance(vv, (int, float, str, bool, type(None)))}
        else:
            out[k] = v
    return out


# ============ System Prompt ============
def _build_system_prompt() -> str:
    tool_desc = get_tool_descriptions()
    return f"""你是一个专业的科研助理 Agent，可以使用以下工具帮助用户：

{tool_desc}

## 工作流程
严格按以下两种 JSON 格式输出，每次只输出一种：

【格式 1：调用工具】
{{"thought": "为什么需要调用这个工具", "action": "工具名", "action_input": {{"参数名": "参数值"}}}}

【格式 2：任务完成】
{{"thought": "我已经收集到足够信息", "final_answer": "给用户的最终答案"}}

## 规则
1. 只输出 JSON，不要输出任何其他文字、markdown 代码块标记
2. 严格区分两种格式：
   - 调用工具用 "action" 键
   - 任务完成用 "final_answer" 键
   注意：final_answer 是一个键，不是工具名！不要写成 {{"action": "final_answer"}}。
3. action 必须是上面列出的工具名之一
4. action_input 的键名必须严格匹配工具签名中的参数名（如 calculator 的参数是 expr）
5. doc_id / doc_ids 参数不要带扩展名，如 'sample' 而不是 'sample.pdf'
6. 每次只调用一个工具
7. 简单问题（打招呼、常识）直接给 final_answer，不要调用工具
8. 涉及论文内容的问题，先调用 rag_search
9. rag_search 的 query 应该包含具体技术术语（如 'Transformer self-attention'、'encoder decoder'），不要用"这篇论文用了什么方法"这种泛问
10. 最多 {MAX_ITER} 轮推理，之后必须给 final_answer
11. final_answer 必须完整、直接回答用户问题，可以引用检索到的内容
12. 工具调用失败或结果不理想时，不要连续调用同一个工具超过 2 次
13. 如果已经调用过 2 次 rag_search，且返回的页码没有新增内容，必须立即给 final_answer
14. 当 rag_search 返回的片段看起来不相关时，不要重复搜同一个 query，而是换具体术语或换工具
15. 任何情况下都必须在 {MAX_ITER} 轮内给出 final_answer
16. 优先用最少轮次完成任务
17. 当用户明确问某篇或多篇文档（如 'sample2 里说了什么'、'对比 sample 和 sample2'）时，
    rag_search 或 paper_compare 要传 doc_ids 参数，如 {{"query": "...", "doc_ids": ["sample2"]}}
"""


# ============ 输出解析器（容错） ============
def _parse_output(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {
        "thought": "解析失败",
        "final_answer": f"模型输出格式错误，无法解析。原始输出：{text[:200]}",
    }


# ============ 工具执行（带容错） ============
def _execute_tool(
    tool_name: str,
    tool_args: dict,
    tools: dict[str, Callable],
    timeout_sec: float = 30.0,
) -> tuple[str, float, bool]:
    t0 = time.time()

    if tool_name not in tools:
        return f"未知工具 {tool_name}。可用工具：{list(tools.keys())}", 0.0, False

    fn = tools[tool_name]
    try:
        result = fn(**tool_args)
        elapsed = (time.time() - t0) * 1000
        return str(result), elapsed, True
    except TypeError as e:
        elapsed = (time.time() - t0) * 1000
        try:
            sig = inspect.signature(fn)
            expected = list(sig.parameters.keys())
            return (f"参数错误：{e}。工具 {tool_name} 的正确参数名是：{expected}",
                    elapsed, False)
        except Exception:
            return f"参数错误：{e}。请检查工具 {tool_name} 的参数。", elapsed, False
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"工具执行异常：{e}", elapsed, False


# ============ 提取 observation 里的页码 ============
_PAGE_RE = re.compile(r"【.+?-第(\d+)页】")


def _extract_pages(observation: str) -> set[int]:
    return set(int(p) for p in _PAGE_RE.findall(observation))


# ============ RAG 命中率 ============
_SCORE_RE = re.compile(r"(?:score\s*[=:]\s*|\[)(\d+\.\d+)")


def _is_rag_hit(observation: str) -> bool:
    scores = [float(s) for s in _SCORE_RE.findall(observation)]
    if not scores:
        return bool(re.search(r"【.+?-第\d+页】", observation))
    return max(scores) >= RAG_HIT_SCORE_THRESHOLD


# ============ 统计：按工具聚合 ============
def _new_tool_stats() -> dict:
    return {
        "_meta": {
            "rag_hit": 0,
            "rag_total": 0,
            "final_tokens": 0,
            "final_calls": 0,
        }
    }


def _record_tool_call(stats: dict, tool_name: str, elapsed_ms: float,
                      success: bool, tokens: int = 0):
    if tool_name not in stats:
        stats[tool_name] = {
            "calls": 0, "success": 0, "fail": 0,
            "total_ms": 0.0, "tokens": 0,
        }
    s = stats[tool_name]
    s["calls"] += 1
    s["total_ms"] += elapsed_ms
    s["tokens"] += tokens
    if success:
        s["success"] += 1
    else:
        s["fail"] += 1


def _record_rag_result(stats: dict, hit: bool):
    stats["_meta"]["rag_total"] += 1
    if hit:
        stats["_meta"]["rag_hit"] += 1


def _record_final(stats: dict, tokens: int):
    stats["_meta"]["final_calls"] += 1
    stats["_meta"]["final_tokens"] += tokens


# ============ 强制生成 final_answer（同步版） ============
def _force_final_answer(
    question: str,
    scratchpad: str,
    observation: str,
    system_prompt: str,
    model: str,
) -> tuple[str | None, int]:
    force_prompt = (
        f"用户问题：{question}\n\n"
        f"已收集的检索结果：\n{scratchpad}\n"
        f"最新 Observation: {observation[:500]}\n\n"
        f"请基于以上信息，直接回答用户问题。"
        f'输出 JSON：{{"thought": "已收集足够信息", "final_answer": "..."}}'
    )
    try:
        resp2 = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": force_prompt},
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 512},
        )
        tokens = (resp2.get("eval_count", 0)
                  + resp2.get("prompt_eval_count", 0))
        out2 = _parse_output(resp2["message"]["content"])
        if "final_answer" in out2 and out2["final_answer"]:
            return str(out2["final_answer"]), tokens
    except Exception:
        pass
    return None, 0


# ============ ReAct 主循环（同步，保留原行为） ============
def react_loop(
    question: str,
    tools: dict | None = None,
    model: str = DEFAULT_MODEL,
    max_iter: int = MAX_ITER,
    verbose: bool = False,
    history: list[dict] | None = None,
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
    """
    history: 多轮对话历史，格式 [{"role": "user"/"assistant", "content": str}, ...]
    doc_ids: 文档范围限定，None = 全部文档
    session_id: 用于日志标记
    """
    if tools is None:
        tools = get_all_tools()

    system_prompt = _build_system_prompt()
    scratchpad = ""
    trace: list[AgentStep] = []
    total_tokens = 0
    overall_t0 = time.time()
    tool_stats = _new_tool_stats()

    seen_pages: set[int] = set()

    history_append = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]

    def _finish(answer: str, success: bool, iterations: int) -> dict:
        history_append[1]["content"] = answer
        elapsed = int((time.time() - overall_t0) * 1000)
        metrics = {
            "total_tokens": total_tokens,
            "iterations": iterations,
            "elapsed_ms": elapsed,
        }
        # 写日志
        _write_log(_build_log_record(
            question=question, answer=answer, trace=trace,
            metrics=metrics, tool_stats=tool_stats,
            history_append=history_append, success=success,
            doc_ids=doc_ids, session_id=session_id, model=model,
        ))
        return {
            "answer": answer,
            "trace": trace,
            "metrics": metrics,
            "tool_stats": tool_stats,
            "success": success,
            "history_append": history_append,
        }

    for i in range(max_iter):
        user_input = f"用户问题：{question}\n\n"
        if scratchpad:
            user_input += f"已完成的步骤：\n{scratchpad}\n\n"
        user_input += "请输出下一步 JSON："

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-6:]:
                messages.append(h)
        messages.append({"role": "user", "content": user_input})

        t0 = time.time()
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                format="json",
                options={"temperature": 0.1, "num_predict": 512},
            )
        except Exception as e:
            return _finish(f"LLM 调用失败：{e}", False, i)

        llm_elapsed = int((time.time() - t0) * 1000)
        tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
        total_tokens += tokens

        out = _parse_output(resp["message"]["content"])
        thought = out.get("thought", "")
        if verbose:
            print(f"\n[Step {i}] Thought: {thought[:100]}")

        if "final_answer" in out and out["final_answer"]:
            _record_final(tool_stats, tokens)
            trace.append(AgentStep(
                step_idx=i, thought=thought, action="Final",
                action_input={}, observation="", elapsed_ms=llm_elapsed,
            ))
            return _finish(str(out["final_answer"]), True, i + 1)

        action = out.get("action")
        args = out.get("action_input", {}) or {}

        if not action:
            scratchpad += f"\n[Step {i}] 输出缺少 action 字段，请重新按格式输出。\n"
            trace.append(AgentStep(
                step_idx=i, thought=thought, action="PARSE_ERROR",
                action_input={}, observation="输出格式错误", elapsed_ms=llm_elapsed,
            ))
            continue

        if not isinstance(args, dict):
            args = {}

        if doc_ids and action in ("rag_search", "paper_compare"):
            if not args.get("doc_ids"):
                args["doc_ids"] = doc_ids

        if verbose:
            print(f"[Step {i}] Action: {action}({args})")

        observation, tool_elapsed, ok = _execute_tool(action, args, tools)
        _record_tool_call(tool_stats, action, tool_elapsed, ok, tokens=tokens)
        if action == "rag_search":
            _record_rag_result(tool_stats, _is_rag_hit(observation))

        if verbose:
            print(f"[Step {i}] Observation: {observation[:150]}")

        if action == "rag_search":
            pages = _extract_pages(observation)
            new_pages = pages - seen_pages
            seen_pages |= pages

            no_new = not new_pages and i >= 1
            too_many = len(seen_pages) >= MAX_TOTAL_PAGES

            if no_new or too_many:
                reason = ("无新增检索结果" if no_new
                          else f"召回页数已达上限 {MAX_TOTAL_PAGES}")

                trace.append(AgentStep(
                    step_idx=i, thought=thought, action=action,
                    action_input=args, observation=observation[:500],
                    elapsed_ms=tool_elapsed,
                ))
                scratchpad += (
                    f"\n[Step {i}]\n"
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
                    f"Observation: {observation[:500]}\n"
                )

                forced, force_tokens = _force_final_answer(
                    question, scratchpad, observation, system_prompt, model,
                )
                total_tokens += force_tokens
                _record_final(tool_stats, force_tokens)
                if forced:
                    trace.append(AgentStep(
                        step_idx=i + 1, thought=f"{reason}，强制终止",
                        action="Final", action_input={}, observation="",
                        elapsed_ms=0,
                    ))
                    return _finish(forced, True, i + 2)

                fallback = f"基于已检索到的信息：\n\n{scratchpad[-800:]}"
                trace.append(AgentStep(
                    step_idx=i + 1, thought=f"{reason}，强制终止（LLM 兜底）",
                    action="Final", action_input={}, observation="",
                    elapsed_ms=0,
                ))
                return _finish(fallback, True, i + 2)

        trace.append(AgentStep(
            step_idx=i, thought=thought, action=action,
            action_input=args, observation=observation[:500],
            elapsed_ms=tool_elapsed,
        ))

        scratchpad += (
            f"\n[Step {i}]\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation: {observation[:500]}\n"
        )

    return _finish(
        f"达到最大迭代次数（{max_iter}），未能完成任务。\n"
        f"已收集的信息：\n{scratchpad[-500:]}",
        False,
        max_iter,
    )


# ============ 流式版本 ============
_STREAM_SYSTEM_PROMPT = """你是一个专业的科研助理。请基于以下已收集的检索结果，直接、完整地回答用户问题。

## 输出格式要求（必须严格遵守）
1. 直接输出答案正文，**不要**输出 JSON、**不要**输出 markdown 代码块标记（```）
2. **每一条来自检索结果的陈述，都必须在句末标注来源**
3. 来源格式**必须**是：【文档名-第X页】
   - 正确示例：Transformer 使用多头注意力机制【sample.pdf-第2页】。
   - 正确示例：该模型在 WMT 2014 英德翻译任务上达到 28.4 BLEU【sample.pdf-第7页】。
   - **错误示例**：Transformer 使用多头注意力机制 [1]。  ← 禁止用 [数字] 格式
   - **错误示例**：Transformer 使用多头注意力机制。  ← 禁止不标来源
4. **不要**照抄检索结果里的 [数字] 引用（如 [17,18]、[9]），必须替换成 【文档名-第X页】
5. 如果检索结果不足以回答，明确说明"知识库中未找到相关内容"
6. 用中文回答，条理清晰

## 用户问题
{question}

## 已收集的检索结果
{scratchpad}
"""


def _stream_final_answer(
    question: str,
    scratchpad: str,
    model: str,
) -> Iterator[str]:
    prompt = _STREAM_SYSTEM_PROMPT.format(
        question=question,
        scratchpad=scratchpad[-2000:] if scratchpad else "（无检索结果）",
    )
    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.1, "num_predict": 512},
        )
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
    except Exception as e:
        yield f"\n\n[生成失败：{e}]"


def react_loop_stream(
    question: str,
    tools: dict | None = None,
    model: str = DEFAULT_MODEL,
    max_iter: int = MAX_ITER,
    history: list[dict] | None = None,
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> Iterator[dict]:
    """
    流式 ReAct 循环。yield 事件流：
      {"type": "thought", "content": str}
      {"type": "action", "tool": str, "args": dict}
      {"type": "observation", "content": str, "elapsed_ms": float}
      {"type": "final_chunk", "content": str}
      {"type": "done", "answer": str, "trace": list, "metrics": dict,
       "tool_stats": dict, "success": bool, "history_append": list}
    """
    if tools is None:
        tools = get_all_tools()

    system_prompt = _build_system_prompt()
    scratchpad = ""
    trace: list[AgentStep] = []
    total_tokens = 0
    overall_t0 = time.time()
    seen_pages: set[int] = set()
    tool_stats = _new_tool_stats()

    history_append = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]

    def _emit_done(answer: str, success: bool, iterations: int) -> dict:
        history_append[1]["content"] = answer
        elapsed = int((time.time() - overall_t0) * 1000)
        metrics = {
            "total_tokens": total_tokens,
            "iterations": iterations,
            "elapsed_ms": elapsed,
        }
        # 写日志
        _write_log(_build_log_record(
            question=question, answer=answer, trace=trace,
            metrics=metrics, tool_stats=tool_stats,
            history_append=history_append, success=success,
            doc_ids=doc_ids, session_id=session_id, model=model,
        ))
        return {
            "type": "done",
            "answer": answer,
            "trace": trace,
            "metrics": metrics,
            "tool_stats": tool_stats,
            "success": success,
            "history_append": history_append,
        }

    def _stream_and_emit_final(thought: str, step_idx: int,
                               final_tokens_hint: int = 0):
        nonlocal total_tokens
        final_step = AgentStep(
            step_idx=step_idx, thought=thought, action="Final",
            action_input={}, observation="", elapsed_ms=0,
        )
        trace.append(final_step)

        collected = []
        for piece in _stream_final_answer(question, scratchpad, model):
            collected.append(piece)
            yield {"type": "final_chunk", "content": piece}

        answer = "".join(collected)
        final_step.observation = answer[:200]

        final_tokens = int(len(answer) * 1.5)
        _record_final(tool_stats, final_tokens + final_tokens_hint)
        total_tokens += final_tokens + final_tokens_hint

        yield _emit_done(answer, True, step_idx + 1)

    for i in range(max_iter):
        user_input = f"用户问题：{question}\n\n"
        if scratchpad:
            user_input += f"已完成的步骤：\n{scratchpad}\n\n"
        user_input += "请输出下一步 JSON："

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history[-6:]:
                messages.append(h)
        messages.append({"role": "user", "content": user_input})

        t0 = time.time()
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                format="json",
                options={"temperature": 0.1, "num_predict": 512},
            )
        except Exception as e:
            yield {"type": "thought", "content": f"LLM 调用失败：{e}"}
            yield _emit_done(f"LLM 调用失败：{e}", False, i)
            return

        llm_elapsed = int((time.time() - t0) * 1000)
        tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
        total_tokens += tokens

        out = _parse_output(resp["message"]["content"])
        thought = out.get("thought", "")
        yield {"type": "thought", "content": thought}

        if "final_answer" in out and out["final_answer"]:
            yield from _stream_and_emit_final(
                thought=thought, step_idx=i, final_tokens_hint=tokens,
            )
            return

        action = out.get("action")
        args = out.get("action_input", {}) or {}

        if not action:
            scratchpad += f"\n[Step {i}] 输出缺少 action 字段，请重新按格式输出。\n"
            yield {"type": "observation", "content": "输出格式错误，缺少 action",
                   "elapsed_ms": 0.0}
            trace.append(AgentStep(
                step_idx=i, thought=thought, action="PARSE_ERROR",
                action_input={}, observation="输出格式错误",
                elapsed_ms=llm_elapsed,
            ))
            continue

        if not isinstance(args, dict):
            args = {}

        if doc_ids and action in ("rag_search", "paper_compare"):
            if not args.get("doc_ids"):
                args["doc_ids"] = doc_ids

        yield {"type": "action", "tool": action, "args": args}

        observation, tool_elapsed, ok = _execute_tool(action, args, tools)
        _record_tool_call(tool_stats, action, tool_elapsed, ok, tokens=tokens)
        if action == "rag_search":
            _record_rag_result(tool_stats, _is_rag_hit(observation))

        yield {"type": "observation", "content": observation[:500],
               "elapsed_ms": tool_elapsed}

        if action == "rag_search":
            pages = _extract_pages(observation)
            new_pages = pages - seen_pages
            seen_pages |= pages

            no_new = not new_pages and i >= 1
            too_many = len(seen_pages) >= MAX_TOTAL_PAGES

            if no_new or too_many:
                reason = ("无新增检索结果" if no_new
                          else f"召回页数已达上限 {MAX_TOTAL_PAGES}")
                trace.append(AgentStep(
                    step_idx=i, thought=thought, action=action,
                    action_input=args, observation=observation[:500],
                    elapsed_ms=tool_elapsed,
                ))
                scratchpad += (
                    f"\n[Step {i}]\n"
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
                    f"Observation: {observation[:500]}\n"
                )
                yield from _stream_and_emit_final(
                    thought=f"{reason}，强制终止", step_idx=i + 1,
                )
                return

        trace.append(AgentStep(
            step_idx=i, thought=thought, action=action,
            action_input=args, observation=observation[:500],
            elapsed_ms=tool_elapsed,
        ))

        scratchpad += (
            f"\n[Step {i}]\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation: {observation[:500]}\n"
        )

    yield from _stream_and_emit_final(
        thought=f"达到最大迭代次数 {max_iter}，兜底输出", step_idx=max_iter,
    )