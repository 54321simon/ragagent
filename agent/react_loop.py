"""手写 ReAct 循环。禁止使用 LangChain Agent 高层封装。

核心流程：
    Thought → Action → Observation → 循环 → Final Answer
"""
import json
import re
import time
from typing import Callable
import ollama

from common.schemas import AgentStep
from tools.registry import get_all_tools, get_tool_descriptions

MAX_ITER = 5
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"


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
5. doc_id 参数不要带 .pdf 后缀，如 'sample' 而不是 'sample.pdf'
6. 每次只调用一个工具
7. 简单问题（打招呼、常识）直接给 final_answer，不要调用工具
8. 涉及论文内容的问题，先调用 rag_search
9. rag_search 的 query 应该包含具体技术术语（如 'Transformer self-attention'、'encoder decoder'），不要用"这篇论文用了什么方法"这种泛问
10. 最多 {MAX_ITER} 轮推理，之后必须给 final_answer
11. final_answer 必须完整、直接回答用户问题，可以引用检索到的内容
12. 工具调用失败或结果不理想时，不要连续调用同一个工具超过 2 次
13. 如果已经调用过 2 次以上 rag_search 仍找不到答案，直接基于已检索到的片段给 final_answer，并在答案中说明"基于现有片段"
14. 当 rag_search 返回的片段看起来不相关时，不要重复搜同一个 query，而是换具体术语或换工具（paper_meta / summarize_paper）
"""


# ============ 输出解析器（容错） ============
def _parse_output(text: str) -> dict:
    """解析 LLM 输出。容错处理：json.loads 失败时正则提取第一个 {...}。"""
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


# ============ 工具执行（带超时 + 容错） ============
def _execute_tool(
    tool_name: str,
    tool_args: dict,
    tools: dict[str, Callable],
    timeout_sec: float = 30.0,
) -> tuple[str, float]:
    """执行工具，返回 (observation, elapsed_ms)。"""
    t0 = time.time()

    if tool_name not in tools:
        return f"未知工具 {tool_name}。可用工具：{list(tools.keys())}", 0.0

    fn = tools[tool_name]
    try:
        result = fn(**tool_args)
        elapsed = (time.time() - t0) * 1000
        return str(result), elapsed
    except TypeError as e:
        elapsed = (time.time() - t0) * 1000
        import inspect
        try:
            sig = inspect.signature(fn)
            expected = list(sig.parameters.keys())
            return (f"参数错误：{e}。工具 {tool_name} 的正确参数名是：{expected}",
                    elapsed)
        except Exception:
            return f"参数错误：{e}。请检查工具 {tool_name} 的参数。", elapsed
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return f"工具执行异常：{e}", elapsed


# ============ ReAct 主循环 ============
def react_loop(
    question: str,
    tools: dict | None = None,
    model: str = DEFAULT_MODEL,
    max_iter: int = MAX_ITER,
    verbose: bool = False,
) -> dict:
    """
    手写 ReAct 循环。

    返回：
    {
        "answer": str,
        "trace": List[AgentStep],
        "metrics": {"total_tokens": int, "iterations": int, "elapsed_ms": int},
        "success": bool,
    }
    """
    if tools is None:
        tools = get_all_tools()

    system_prompt = _build_system_prompt()
    scratchpad = ""
    trace: list[AgentStep] = []
    total_tokens = 0
    overall_t0 = time.time()

    for i in range(max_iter):
        user_input = f"用户问题：{question}\n\n"
        if scratchpad:
            user_input += f"已完成的步骤：\n{scratchpad}\n\n"
        user_input += "请输出下一步 JSON："

        t0 = time.time()
        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                format="json",
                options={"temperature": 0.1, "num_predict": 512},
            )
        except Exception as e:
            elapsed = int((time.time() - overall_t0) * 1000)
            return {
                "answer": f"LLM 调用失败：{e}",
                "trace": trace,
                "metrics": {"total_tokens": total_tokens, "iterations": i,
                            "elapsed_ms": elapsed},
                "success": False,
            }

        llm_elapsed = int((time.time() - t0) * 1000)
        tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
        total_tokens += tokens

        out = _parse_output(resp["message"]["content"])
        thought = out.get("thought", "")
        if verbose:
            print(f"\n[Step {i}] Thought: {thought[:100]}")

        # 终止条件：final_answer
        if "final_answer" in out and out["final_answer"]:
            trace.append(AgentStep(
                step_idx=i,
                thought=thought,
                action="Final",
                action_input={},
                observation="",
                elapsed_ms=llm_elapsed,
            ))
            elapsed = int((time.time() - overall_t0) * 1000)
            return {
                "answer": str(out["final_answer"]),
                "trace": trace,
                "metrics": {"total_tokens": total_tokens,
                            "iterations": i + 1, "elapsed_ms": elapsed},
                "success": True,
            }

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

        if verbose:
            print(f"[Step {i}] Action: {action}({args})")

        observation, tool_elapsed = _execute_tool(action, args, tools)

        if verbose:
            print(f"[Step {i}] Observation: {observation[:150]}")

        trace.append(AgentStep(
            step_idx=i,
            thought=thought,
            action=action,
            action_input=args,
            observation=observation[:500],
            elapsed_ms=tool_elapsed,
        ))

        scratchpad += (
            f"\n[Step {i}]\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation: {observation[:500]}\n"
        )

    elapsed = int((time.time() - overall_t0) * 1000)
    return {
        "answer": f"达到最大迭代次数（{max_iter}），未能完成任务。\n"
                  f"已收集的信息：\n{scratchpad[-500:]}",
        "trace": trace,
        "metrics": {"total_tokens": total_tokens,
                    "iterations": max_iter, "elapsed_ms": elapsed},
        "success": False,
    }