"""测试 reranker 打分。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.api import retrieve_hybrid
from retriever.retrievers.reranker import rerank

cands = retrieve_hybrid("Transformer 方法", topk=10)
cands_dict = [
    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_name": c.doc_name,
     "page": c.page, "text": c.text, "score": c.score}
    for c in cands
]

ranked = rerank("Transformer 方法", cands_dict, topk=3)
print("rerank 后 top 3:")
for r in ranked:
    print(f'  {r["chunk_id"]} | rerank={r["rerank_score"]:.4f}')