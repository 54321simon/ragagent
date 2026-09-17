from retriever.api import ingest_pdf, retrieve, get_docs

# 1. 入库
n = ingest_pdf("./data/papers/sample.pdf")
print(f"入库 {n} 个 chunk")

# 2. 查看文档列表
print("文档列表：", get_docs())

# 3. 检索
print("\n检索「这篇论文里RAG有哪些主流范式？」：")
for r in retrieve("这篇论文里RAG有哪些主流范式？", topk=5):
    print(f"[{r.score:.3f}] {r.cite()}  {r.text[:60]}...")