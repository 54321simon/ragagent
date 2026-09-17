# verify_w2.ps1 —— W2 检索层扩展验证
$ErrorActionPreference = "Stop"

Write-Host "`n=== Step 1: 多格式加载 ===" -ForegroundColor Green
uv run python -c "import sys; sys.path.insert(0, r'D:\git大作业'); import docx; print('docx ok')"

Write-Host "`n=== Step 2: 递归切分 ===" -ForegroundColor Green
uv run python -c "
import sys; sys.path.insert(0, r'D:\git大作业')
from retriever.loader import load_pdf
from retriever.chunker import chunk_pages, chunk_pages_recursive
p = load_pdf('data/papers/sample.pdf')
print(f'固定: {len(chunk_pages(p))} 递归: {len(chunk_pages_recursive(p))}')
"

Write-Host "`n=== Step 3: BM25 ===" -ForegroundColor Green
uv run python -c "
import sys; sys.path.insert(0, r'D:\git大作业')
from retriever.loader import load_pdf
from retriever.chunker import chunk_pages
from retriever.retrievers.bm25 import BM25Retriever
p = load_pdf('data/papers/sample.pdf')
r = BM25Retriever(chunk_pages(p))
print(f'BM25 检索到 {len(r.search(\"attention\", 3))} 个')
"

Write-Host "`n=== Step 4: RRF ===" -ForegroundColor Green
uv run python -c "
import sys; sys.path.insert(0, r'D:\git大作业')
from retriever.retrievers.rrf import rrf_fusion
a = [{'chunk_id': 'c1'}, {'chunk_id': 'c2'}]
b = [{'chunk_id': 'c2'}, {'chunk_id': 'c3'}]
r = rrf_fusion(a, b, topk=3)
print(f'RRF 融合 {len(r)} 个，top1={r[0][\"chunk_id\"]}')
"

Write-Host "`n=== Step 5: 混合检索 ===" -ForegroundColor Green
uv run python -c "
import sys; sys.path.insert(0, r'D:\git大作业')
from retriever.api import retrieve_hybrid
r = retrieve_hybrid('attention', topk=3)
print(f'混合检索 {len(r)} 个')
"

Write-Host "`n=== Step 6: 三档对比实验 ===" -ForegroundColor Green
uv run python -c "import sys; sys.path.insert(0, r'D:\git大作业'); exec(open('experiments/retrieval_exp.py', encoding='utf-8').read())"

Write-Host "`n=== W2 检索层全部通过 ===" -ForegroundColor Yellow