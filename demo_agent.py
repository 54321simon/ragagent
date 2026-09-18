"""ReAct Agent 验证。"""
from agent.react_loop import react_loop


def run_case(name: str, question: str):
    print("\n" + "=" * 70)
    print(f"测试: {name}")
    print(f"问题: {question}")
    print("=" * 70)

    result = react_loop(question, verbose=True)

    print(f"\n--- 最终答案 ---")
    print(result["answer"])
    print(f"\n--- 指标 ---")
    print(f"迭代轮次: {result['metrics']['iterations']}")
    print(f"总 token: {result['metrics']['total_tokens']}")
    print(f"总耗时: {result['metrics']['elapsed_ms']}ms")
    print(f"成功: {result['success']}")

    print(f"\n--- 推理轨迹 ---")
    for s in result["trace"]:
        print(f"  Step {s.step_idx}: [{s.action}] {s.thought[:60]}")
        if s.action_input:
            print(f"    input: {s.action_input}")
        if s.observation:
            print(f"    obs: {s.observation[:80]}")


if __name__ == "__main__":
    # 测试 1：简单问题，应该直接回答
    run_case("简单常识", "你好，请介绍一下你自己")

    # 测试 2：计算题，应该调 calculator
    run_case("计算", "3.14 乘以 2.56 等于多少？")

    # 测试 3：查论文，应该调 rag_search
    run_case("论文检索", "这篇论文用了什么方法？")

    # 测试 4：列出文档，应该调 list_documents
    run_case("列文档", "知识库里有哪些论文？")