"""手写 ReAct 循环。禁止使用 LangChain Agent 高层封装。

核心流程：
    Thought → Action → Observation → 循环 → Final Answer
支持多轮对话（history 参数）和文档范围限定（doc_ids 参数）。
内置两层硬性保护：
  1. rag_search 本轮无新增页码 → 强制 final_answer
  2. rag_search 累计召回页数 >= MAX_TOTAL_PAGES → 强制 final_answer

本文件同时提供：
  - react_loop          ：同步阻塞版（评测脚本 / demo_agent.py 使用）
  - react_loop_stream   ：流式事件版（Streamlit UI 使用）

并行工具调用：
  - LLM 可输出 {"actions": [{"action": "...", "action_input": {...}}, ...]}
  - 最多 MAX_PARALLEL_ACTIONS 个，用 ThreadPoolExecutor 并行执行
  - 只在 react_loop_stream 里支持；同步版保持单工具

历史管理：
  - estimate_tokens        ：粗估文本 token
  - summarize_history      ：把旧历史压缩成摘要
  - trim_history           ：主入口，返回 (新历史, 新摘要)

日志：
  - 每次对话结束后写入 data/logs/agent_YYYYMMDD.jsonl
  - 可通过环境变量 LOG_DIR 覆盖目录，LOG_ENABLED=0 关闭
"""
import json
import os
import re
import time
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Iterator
import ollama

from common.schemas import AgentStep
from tools.registry import get_all_tools, get_tool_descriptions

MAX_ITER = 4
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
MAX_TOTAL_PAGES = 5
RAG_HIT_SCORE_THRESHOLD = 0.5

# ============ 并行工具调用 ============
MAX_PARALLEL_ACTIONS = 3        # 单次最多并行 3 个工具

# ============ 历史管理参数 ============
MAX_HISTORY_TOKENS = 3000
KEEP_RECENT_TURNS = 3
SUMMARY_MAX_TOKENS = 300


# ============ 日志 ============
_LOG_DIR = os.environ.get(
    "LOG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "logs"),
)
_LOG_ENABLED = os.environ.get("LOG_ENABLED", "1") != "0"


def _step_to_dict(step) -> dict:
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
    if not _LOG_ENABLED:
        return
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fname = f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"
        path = os.path.join(_LOG_DIR, fname)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_log_record(
    question: str, answer: str, trace: list, metrics: dict,
    tool_stats: dict, history_append: list, success: bool,
    doc_ids: list | None, session_id: str | None, model: str,
    extra: dict | None = None,
) -> dict:
    citations = []
    for m in re.finditer(r"【([^】]+?)-第(\d+)页】", answer or ""):
        citations.append({"doc_name": m.group(1), "page": int(m.group(2))})
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


# ============ 历史管理 ============
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text)


def _history_tokens(history: list[dict]) -> int:
    if not history:
        return 0
    total = 0
    for h in history:
        total += estimate_tokens(h.get("content", ""))
    return total


def summarize_history(
    history: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    if not history:
        return ""

    lines = []
    for h in history:
        role = "用户" if h["role"] == "user" else "助手"
        lines.append(f"{role}：{h.get('content', '')}")
    dialog = "\n".join(lines)

    prompt = (
        "请把下面这段多轮对话压缩成一段简短的摘要，保留关键事实、"
        "用户意图、已讨论的主题和结论，不要遗漏重要信息，不要编造。\n\n"
        f"对话内容：\n{dialog}\n\n"
        "请直接输出摘要正文，不要加任何前缀、标题或代码块标记。"
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": SUMMARY_MAX_TOKENS},
        )
        return resp["message"]["content"].strip()
    except Exception:
        return dialog[-1000:]


def trim_history(
    history: list[dict],
    existing_summary: str = "",
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], str]:
    if not history:
        return [], existing_summary

    total_tokens = _history_tokens(history)
    if total_tokens <= MAX_HISTORY_TOKENS:
        return history, existing_summary

    keep_n = KEEP_RECENT_TURNS * 2
    if len(history) <= keep_n:
        return history[-keep_n:], existing_summary

    old_part = history[:-keep_n]
    recent_part = history[-keep_n:]

    combined = []
    if existing_summary:
        combined.append({"role": "system",
                         "content": f"[之前的摘要] {existing_summary}"})
    combined.extend(old_part)

    new_summary = summarize_history(combined, model=model)
    return recent_part, new_summary


# ============ System Prompt ============
def _build_system_prompt() -> str:
    tool_desc = get_tool_descriptions()
    return f"""你是一个专业的科研助理 Agent，可以使用以下工具帮助用户：

{tool_desc}

## 工作流程
严格按以下三种 JSON 格式输出，每次只输出一种：

【格式 1：调用单个工具】
{{"thought": "为什么需要调用这个工具", "action": "工具名", "action_input": {{"参数名": "参数值"}}}}

【格式 2：并行调用多个独立工具】
{{"thought": "为什么需要同时调用这些工具", "actions": [
    {{"action": "工具名1", "action_input": {{"参数名": "参数值"}}}},
    {{"action": "工具名2", "action_input": {{"参数名": "参数值"}}}}
]}}

【格式 3：任务完成】
{{"thought": "我已经收集到足够信息", "final_answer": "给用户的最终答案"}}

## 规则
1. 只输出 JSON，不要输出任何其他文字、markdown 代码块标记
2. 严格区分三种格式：
   - 调用单个工具用 "action" 键
   - 调用多个工具用 "actions" 键（数组）
   - 任务完成用 "final_answer" 键
   注意：final_answer 是一个键，不是工具名！不要写成 {{"action": "final_answer"}}。
3. action 必须是上面列出的工具名之一
4. action_input 的键名必须严格匹配工具签名中的参数名（如 calculator 的参数是 expr）
5. doc_id / doc_ids 参数不要带扩展名，且**必须使用知识库中真实的 doc_id**
6. **重要：任何涉及具体论文的 doc_id 都必须先通过 list_documents 获取，禁止编造！**
   - 如果用户说"这篇论文/两篇论文/这些论文"但没指明具体文档 → 必须先调 list_documents
   - 如果用户明确说了文档名，也要先调 list_documents 确认真实的 doc_id
   - **绝对禁止使用 'sample'、'sample2'、'论文A'、'论文B' 这种示例值**，它们不是真实 doc_id
7. **只有在多个工具之间没有依赖关系时才用 "actions" 并行调用**，最多 {MAX_PARALLEL_ACTIONS} 个
   - 适合并行：同时查多篇论文的元信息、同时搜多个不同关键词
   - 不适合并行：需要上一个工具的结果作为下一个的输入
8. 简单问题（打招呼、常识）直接给 final_answer，不要调用工具
9. 涉及论文内容的问题，先调用 rag_search
10. rag_search 的 query 应该包含具体技术术语，不要用"这篇论文用了什么方法"这种泛问
11. 最多 {MAX_ITER} 轮推理，之后必须给 final_answer
12. final_answer 必须完整、直接回答用户问题，可以引用检索到的内容
13. 工具调用失败或结果不理想时，不要连续调用同一个工具超过 2 次
14. 如果 rag_search 返回"doc_id 不存在，请用下列真实 doc_id"，立即改用返回的真实 doc_id 重试
15. 如果已经调用过 2 次 rag_search，且返回的页码没有新增内容，必须立即给 final_answer
16. 任何情况下都必须在 {MAX_ITER} 轮内给出 final_answer
17. 优先用最少轮次完成任务
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


# ============ 构造 messages ============
def _build_messages(
    system_prompt: str,
    user_input: str,
    history: list[dict] | None,
    history_summary: str = "",
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if history_summary:
        messages.append({
            "role": "system",
            "content": f"[历史对话摘要]\n{history_summary}",
        })
    if history:
        for h in history[-6:]:
            messages.append(h)
    messages.append({"role": "user", "content": user_input})
    return messages


# ============ 并行执行工具 ============
def _execute_tools_parallel(
    items: list[dict],
    tools: dict[str, Callable],
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    并行执行多个工具。每个 item 是 {"action": str, "action_input": dict}。
    返回 list[dict]，每项含：
        {"tool": str, "args": dict, "observation": str,
         "elapsed_ms": float, "ok": bool}
    顺序与输入 items 一致。
    """
    # 规范化参数
    normalized = []
    for it in items:
        action = it.get("action")
        args = it.get("action_input", {}) or {}
        if not isinstance(args, dict):
            args = {}
        if doc_ids and action in ("rag_search", "paper_compare"):
            if not args.get("doc_ids"):
                args["doc_ids"] = doc_ids
        normalized.append({"action": action, "args": args})

    results = [None] * len(normalized)

    def _run(idx: int, action: str, args: dict):
        obs, ms, ok = _execute_tool(action, args, tools)
        return idx, {"tool": action, "args": args, "observation": obs,
                     "elapsed_ms": ms, "ok": ok}

    with ThreadPoolExecutor(max_workers=min(len(normalized),
                                             MAX_PARALLEL_ACTIONS)) as ex:
        futures = [
            ex.submit(_run, i, n["action"], n["args"])
            for i, n in enumerate(normalized)
        ]
        for fut in as_completed(futures):
            try:
                idx, res = fut.result()
                results[idx] = res
            except Exception as e:
                # 极端情况（不会发生，_execute_tool 自己 catch 了）
                pass

    # 兜底：如果某个位置没填上，补一个错误结果
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "tool": normalized[i]["action"],
                "args": normalized[i]["args"],
                "observation": "并行执行异常",
                "elapsed_ms": 0.0,
                "ok": False,
            }
    return results


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


# ============ ReAct 主循环（同步，保持单工具） ============
def react_loop(
    question: str,
    tools: dict | None = None,
    model: str = DEFAULT_MODEL,
    max_iter: int = MAX_ITER,
    verbose: bool = False,
    history: list[dict] | None = None,
    history_summary: str = "",
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
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

        messages = _build_messages(
            system_prompt, user_input, history, history_summary,
        )

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

        # 同步版：如果 LLM 输出了 actions，退化为取第一个
        if "actions" in out and isinstance(out["actions"], list) and out["actions"]:
            first = out["actions"][0] or {}
            out["action"] = first.get("action")
            out["action_input"] = first.get("action_input", {})

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
    history_summary: str = "",
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> Iterator[dict]:
    """
    流式 ReAct 循环。yield 事件流：
      {"type": "thought", "content": str}
      {"type": "action", "tool": str, "args": dict}                 # 单工具
      {"type": "actions", "items": [{"tool": str, "args": dict}, ...]}  # 并行
      {"type": "observation", "content": str, "elapsed_ms": float}  # 单工具结果
      {"type": "action_result", "tool": str, "observation": str,
       "elapsed_ms": float, "ok": bool}                              # 并行时单个完成
      {"type": "observations", "items": [...]}                       # 并行全部完成
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

    def _handle_single_action(action: str, args: dict, thought: str,
                              step_idx: int, tokens: int,
                              llm_elapsed: int):
        """处理单工具分支。yield 事件；返回 (action, observation, ok,
        tool_elapsed) 或 None（表示已 yield final）。"""
        if doc_ids and action in ("rag_search", "paper_compare"):
            if not args.get("doc_ids"):
                args["doc_ids"] = doc_ids

        yield {"type": "action", "tool": action, "args": args}, None

        observation, tool_elapsed, ok = _execute_tool(action, args, tools)
        _record_tool_call(tool_stats, action, tool_elapsed, ok, tokens=tokens)
        if action == "rag_search":
            _record_rag_result(tool_stats, _is_rag_hit(observation))

        yield {"type": "observation", "content": observation[:500],
               "elapsed_ms": tool_elapsed}, None

        yield None, (action, args, observation, tool_elapsed, ok)

    for i in range(max_iter):
        user_input = f"用户问题：{question}\n\n"
        if scratchpad:
            user_input += f"已完成的步骤：\n{scratchpad}\n\n"
        user_input += "请输出下一步 JSON："

        messages = _build_messages(
            system_prompt, user_input, history, history_summary,
        )

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

        # ---- 分支 1：final_answer ----
        if "final_answer" in out and out["final_answer"]:
            yield from _stream_and_emit_final(
                thought=thought, step_idx=i, final_tokens_hint=tokens,
            )
            return

        # ---- 分支 2：并行 actions ----
        if "actions" in out and isinstance(out["actions"], list) and out["actions"]:
            items = out["actions"][:MAX_PARALLEL_ACTIONS]
            norm_items = []
            for it in items:
                a = it.get("action")
                ia = it.get("action_input", {}) or {}
                if not isinstance(ia, dict):
                    ia = {}
                if not a:
                    continue
                if doc_ids and a in ("rag_search", "paper_compare"):
                    if not ia.get("doc_ids"):
                        ia["doc_ids"] = doc_ids
                norm_items.append({"tool": a, "args": ia})

            if not norm_items:
                scratchpad += f"\n[Step {i}] actions 数组为空或格式错误。\n"
                yield {"type": "observation",
                       "content": "actions 数组为空或格式错误",
                       "elapsed_ms": 0.0}
                continue

            yield {"type": "actions", "items": norm_items}

            # 并行执行（内部不 yield，一次性返回所有结果）
            raw = _execute_tools_parallel(
                [{"action": n["tool"], "action_input": n["args"]}
                 for n in norm_items],
                tools, doc_ids=None,   # doc_ids 上面已经注入
            )

            # 逐个 yield action_result
            observations_text = []
            for r in raw:
                _record_tool_call(tool_stats, r["tool"], r["elapsed_ms"],
                                  r["ok"], tokens=tokens)
                if r["tool"] == "rag_search":
                    _record_rag_result(tool_stats, _is_rag_hit(r["observation"]))
                yield {"type": "action_result",
                       "tool": r["tool"],
                       "observation": r["observation"][:500],
                       "elapsed_ms": r["elapsed_ms"],
                       "ok": r["ok"]}

            # 拼合并 observation
            merged_parts = []
            for r in raw:
                status = "✓" if r["ok"] else "✗"
                merged_parts.append(
                    f"[{status} {r['tool']}] "
                    f"{r['observation'][:500]}"
                )
            merged = "\n\n".join(merged_parts)

            yield {"type": "observations",
                   "items": [{"tool": r["tool"],
                              "observation": r["observation"][:500],
                              "elapsed_ms": r["elapsed_ms"],
                              "ok": r["ok"]} for r in raw]}

            # 记 trace（每个工具一条）
            for r in raw:
                trace.append(AgentStep(
                    step_idx=i, thought=thought,
                    action=r["tool"], action_input=r["args"],
                    observation=r["observation"][:500],
                    elapsed_ms=r["elapsed_ms"],
                ))

            # 更新 scratchpad
            scratchpad += f"\n[Step {i} - 并行调用]\nThought: {thought}\n"
            for r in raw:
                scratchpad += (
                    f"Action: {r['tool']}\n"
                    f"Action Input: {json.dumps(r['args'], ensure_ascii=False)}\n"
                    f"Observation: {r['observation'][:500]}\n"
                )

            # 硬性保护：如果并行里有 rag_search 且无新增页码
            any_rag = any(r["tool"] == "rag_search" for r in raw)
            if any_rag:
                for r in raw:
                    if r["tool"] == "rag_search":
                        pages = _extract_pages(r["observation"])
                        new_pages = pages - seen_pages
                        seen_pages |= pages
                too_many = len(seen_pages) >= MAX_TOTAL_PAGES
                if too_many:
                    yield from _stream_and_emit_final(
                        thought=f"召回页数已达上限 {MAX_TOTAL_PAGES}，强制终止",
                        step_idx=i + 1,
                    )
                    return

            continue   # 进入下一轮

        # ---- 分支 3：单工具 action ----
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

        # 用生成器实现单工具流程
        gen = _handle_single_action(action, args, thought, i, tokens, llm_elapsed)
        result = None
        for event, ret in gen:
            if event is not None:
                yield event
            if ret is not None:
                result = ret

        if result is None:
            continue
        action, args, observation, tool_elapsed, ok = result

        # 硬性保护
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