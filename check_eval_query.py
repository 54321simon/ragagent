import sys
sys.path.insert(0, r'D:\git大作业')
from retriever.api import retrieve, retrieve_hybrid

eval_questions = [
    "RAG最优集成策略是什么？",
    "RAG相关的评估指标有哪些？",
    "RAG输出质量会被哪些因素影响？"
]

for q in eval_questions:
    print(f"\n==== 问题：{q} ====")
    print("【纯向量检索top5】")
    vec_res = retrieve(q, topk=5)
    for item in vec_res:
        print(f"  chunk_id:{item.chunk_id}, score:{item.score:.4f}")

    print("【混合检索top5】")
    hyb_res = retrieve_hybrid(q, topk=5)
    for item in hyb_res:
        print(f"  chunk_id:{item.chunk_id}, score:{item.score:.4f}")