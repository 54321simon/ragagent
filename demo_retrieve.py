from retriever.api import (
    get_docs,
    retrieve, retrieve_hybrid, retrieve_hybrid_rerank,
)

print("文档列表：", get_docs())

queries = [
    "这篇论文提出的模型架构叫什么？",
    "多头注意力机制是怎么计算的？",
    "位置编码用了什么函数？",
    "训练用了什么优化器？",
    "BLEU 分数是多少？",
    "self-attention 的计算复杂度是多少？",
    "这篇论文里RAG有哪些主流范式？",
]

modes = [
    ("vector", retrieve),
    ("hybrid", retrieve_hybrid),
    ("hybrid_rerank", retrieve_hybrid_rerank),
]

for mode_name, fn in modes:
    print(f"\n{'=' * 60}")
    print(f"模式：{mode_name}")
    print('=' * 60)
    for q in queries:
        try:
            results = fn(q, topk=5)
        except Exception as e:
            print(f"「{q}」→ 异常: {type(e).__name__}: {e}")
            continue
        if not results:
            print(f"「{q}」→ 无结果")
            continue
        top = results[0]
        print(f"「{q}」")
        print(f"  top-1 [{top.score:.3f}] {top.cite()}")
        print(f"        {top.text[:70]}...")