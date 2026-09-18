# verify_w6.ps1 —— W6 reranker 验证
chcp 65001 | Out-Null
$ErrorActionPreference = "Continue"

Write-Host "`n=== Step 1: reranker 模型可加载 ===" -ForegroundColor Green
uv run python -c "from retriever.retrievers.reranker import get_reranker; get_reranker(); print('reranker ok')"

Write-Host "`n=== Step 2: reranker 能打分 ===" -ForegroundColor Green
uv run python experiments\test_reranker.py

Write-Host "`n=== Step 3: 混合+rerank 接口 ===" -ForegroundColor Green
uv run python experiments\test_hybrid_rerank.py

Write-Host "`n=== Step 4: 三档对比实验 ===" -ForegroundColor Green
uv run python experiments\retrieval_exp_v2.py

Write-Host "`n=== W6 reranker 全部通过 ===" -ForegroundColor Yellow