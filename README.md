# 智能科研助理 题目一

基于原仓库 https://github.com/54321simon/ragagent 继续完成的本地 RAG + 手写 ReAct 系统。12 篇计算机视觉与 Transformer 论文、68 道检索评测题、24 道真实模型 Agent 评测题。结果在 experiments/results，报告在 docs。

## 本机启动

双击 start_app.cmd，打开 http://127.0.0.1:8501。代码和 Python 环境在 D:/生产实习大作业/ragagent，模型在同级 models，索引在 D:/NJAU_RAG_Data/index。需要保持这些目录。首次模型加载及 CPU 生成较慢。

## 新电脑部署

1. 将仓库解压到 D 盘。在 PowerShell 执行 `powershell -ExecutionPolicy Bypass -File scripts/setup.ps1`。首次安装需要联网，自动下载官方 uv、Python、Ollama 和两个模型。
2. 检查 .env 中 INDEX_DIR 使用 ASCII 路径。不同项目应使用不同索引目录。
3. 使用 `.venv/Scripts/python.exe scripts/download_papers.py` 下载论文，或保留交付包内的 data/papers。
4. 使用 `.venv/Scripts/python.exe scripts/ingest_corpus.py` 创建索引。首次完整嵌入需要几分钟。
5. 双击 start_app.cmd。

默认 vector 是本次 60 道题上 Hit@5 最好的模式。修改 .env 的 RETRIEVAL_MODE 为 hybrid 或 rerank 可以切换；更换配置后重启应用。Rerank 需下载 BAAI/bge-reranker-base，本机使用同级 models/bge-reranker-base。更换嵌入模型必须新建索引，不能混用不同向量。

## 复现测试与实验

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe experiments/evaluate.py --modes vector hybrid rerank
.venv/Scripts/python.exe experiments/component_exp.py
.venv/Scripts/python.exe experiments/embedding_exp.py
.venv/Scripts/python.exe experiments/prompt_exp.py
.venv/Scripts/python.exe experiments/agent_eval.py
.venv/Scripts/python.exe experiments/robustness.py
```

embedding_exp 需要同级 models 下的 m3e-base 和 bge-large-zh-v1.5（本机已下载）。全部数据使用真实模型，本机 CPU 延迟受并行实验影响。组件实验仅 2 篇论文、8 道题，不应外推到全语料。manual_review.csv 保留独立人工评分列，不用执行成功率替代正确性。

## Docker

安装 Docker 后在仓库执行 `docker compose up --build`。初始化服务下载模型，首次随后执行 `docker compose exec app python scripts/ingest_corpus.py`，打开 localhost:8501。本机无 Docker，配置只经过静态检查，未声明镜像构建或容器运行通过。默认仅绑定本机端口。

## 工程实现

retriever：多格式加载、四类切分、Chroma 增量索引、向量/BM25/RRF/重排序。
rag：证据拼接、页码引用、语义缓存、降级、自检、请求日志。
agent：共享同步/流式 ReAct 引擎、快速路由、最多三工具并行、错误恢复、历史窗口和摘要。
tools：8 个注册工具；联网学术检索为可选扩展，当前不提供联网搜索。
ui：三列 Streamlit、文档管理、会话隔离、过程轨迹、指标与健康检查。

PDF 页码采用物理页，不是期刊页码。DOCX/TXT/MD 当前标注为逻辑页 1。扫描版 PDF 无 OCR。模型可能误读证据；缺少逐句引用时显示核对提示，错误页码替换为“引用未核验”，不缓存这类答案。引用存在只证明来源可定位，不能证明句子事实正确。

## 提交前

填写报告封面的组员、学号、班级和教师信息。由独立评审人在 manual_review.csv 填写正确性、完整性、引用准确性。答辩可按 PPT 备注演示。完整历史保留在 Git 中，完成版本已推送到 codex/coursework-completion 分支。

GitHub 入口：https://github.com/54321simon/ragagent/tree/codex/coursework-completion 。本机账号 ccc164184 已取得该仓库写入权限。
