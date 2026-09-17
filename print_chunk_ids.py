import sys
sys.path.insert(0, r'D:\git大作业')
from retriever.api import ingest_pdf, retrieve_hybrid

# 清空旧索引，重新入库
import os
import shutil
index_path = "data/index"
if os.path.exists(index_path):
    shutil.rmtree(index_path)

n = ingest_pdf('data/papers/sample.pdf')
print(f"入库chunk总数：{n}")
res = retrieve_hybrid("注意力机制是什么？", topk=5)
print("\n=== 检索结果chunk信息 ===")
for idx, r in enumerate(res):
    print(f"{idx+1}. chunk_id={r.chunk_id} | doc_name={r.doc_name} | page={r.page}")
    print(f"text片段：{r.text[:80]}\n")