"""测试混合+rerank 接口。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.api import retrieve_hybrid_rerank

res = retrieve_hybrid_rerank("Transformer 方法", topk=3)
print(f"检索到 {len(res)} 个")
for r in res:
    print(f'  [{r.score:.4f}] {r.cite()} {r.text[:60]}')