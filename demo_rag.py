"""RAG 生成层验证。"""
from rag.chain import rag_answer, rag_answer_stream


def test_sync():
    print("=" * 60)
    print("测试 1：同步 RAG")
    print("=" * 60)

    q = "这篇论文用了什么方法？"
    print(f"问题：{q}\n")
    r = rag_answer(q)

    print("答案：")
    print(r["answer"])
    print()
    print("引用：")
    for c in r["citations"]:
        print(f"  {c.cite()}  (score={c.score:.4f})")
    print()
    print("是否降级：", r["degraded"])
    # 只有非降级才打印metrics和debug
    if not r["degraded"]:
        print("指标：", r["metrics"])
        print("引用校验：", {
            "实际引用": len(r["debug"]["cited"]),
            "幻觉引用": r["debug"]["hallucinated"],
            "未用片段": len(r["debug"]["unused"]),
        })
    else:
        print("降级类型：", r["degrade_type"])


def test_stream():
    print("\n" + "=" * 60)
    print("测试 2：流式 RAG")
    print("=" * 60)

    q = "论文的核心贡献是什么？"
    print(f"问题：{q}\n")
    print("答案（流式）：")
    for token in rag_answer_stream(q):
        print(token, end="", flush=True)
    print("\n")


def test_no_results():
    print("=" * 60)
    print("测试 3：知识库无相关内容（降级 1）")
    print("=" * 60)

    q = "如何做红烧肉？"   # 故意问和论文无关的
    r = rag_answer(q)
    print(f"问题：{q}")
    print("答案：", r["answer"][:200])
    print("是否降级：", r["degraded"], "| 类型：", r.get("degrade_type"))


if __name__ == "__main__":
    test_sync()
    test_stream()
    test_no_results()