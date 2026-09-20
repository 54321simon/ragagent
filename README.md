# 智能科研助理

> 基于 RAG + Agent 的论文知识库问答系统
>
> 南京农业大学课程实践项目

## 一、项目简介

上传论文 PDF/Word，用自然语言提问，Agent 自动推理并给出**带引用来源**的答案。

**核心特性**：

- 🏠 完全本地部署：Ollama + bge-m3，不依赖云端 API
- 🔧 手写 ReAct 循环：约 250 行，不用 LangChain Agent 高层封装
- 🔍 混合检索：bge-m3 向量 + BM25 + RRF + 可选 reranker
- 📚 引用溯源：答案标注 【文档名-第X页】
- 🔬 可视化推理：右侧展示 Thought → Action → Observation
- 🎯 多文档检索：可选 1 篇或多篇文档提问
- 🛡️ LLM 自检拒答：库外问题不硬凑

## 二、快速开始

### 环境要求

- Windows 10/11 或 Linux
- Python 3.10+
- uv 包管理器
- Ollama 本地大模型服务

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/54321simon/ragagent.git
cd ragagent
```

2. 安装依赖

```bash
uv sync
```

3. 拉取 Ollama 模型

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull bge-m3:567m
```

4. 配置环境变量（可选）

```bash
cp .env.example .env
```

5. 启动应用

```bash
uv run streamlit run ui/app.py
```

浏览器打开 http://localhost:8501

## 三、项目结构

```
agent/          Agent 决策层（手写 ReAct 循环）
common/         共享数据结构
rag/            RAG 生成层（prompt / context / citation / fallback / self_check）
retriever/      检索层（loader / chunker / embedder / store / api + BM25/RRF/reranker）
tools/          Agent 工具集（8 个工具）
ui/             Streamlit 前端
experiments/    评测脚本 + 20 条评测集
data/           论文文件 + Chroma 索引（gitignore）
```

## 四、技术栈

- LLM: Ollama + qwen2.5:7b-instruct-q4_K_M
- Embedding: bge-m3:567m（1024 维）
- Reranker: BAAI/bge-reranker-base
- 向量库: ChromaDB（cosine）
- 关键词检索: rank_bm25 + jieba
- 融合: 手写 RRF
- 前端: Streamlit
- 环境: Python 3.10 + uv

## 五、核心设计

### 5.1 手写 ReAct 循环

- System Prompt 用 inspect.signature 自动生成工具签名
- 强制 JSON 输出：ollama.chat(..., format="json")
- 三层容错解析：json.loads → 正则提取 → 兜底
- 硬性保护：rag_search 无新增页码时强制终止
- 最多 4 轮

### 5.2 三档检索对比（15 条可回答 query）

| 档位 | Hit@1 | Hit@5 | MRR | 延迟(ms) |
|---|---|---|---|---|
| vector | 0.667 | 0.800 | 0.717 | 42 |
| hybrid | 0.667 | 0.867 | 0.739 | 44 |
| hybrid_rerank | 0.667 | 0.867 | 0.767 | 1592 |

结论：hybrid 是默认档。

### 5.3 8 个工具

rag_search / paper_meta / paper_compare / extract_keywords / summarize_paper / current_time / calculator / list_documents

### 5.4 拒答机制

两层保护：

1. LLM 自检（rag/self_check.py）
2. Agent 决策判断

## 六、评测

```bash
uv run python experiments/retrieval_exp_v2.py
```

## 七、已知问题

1. DOCX 切分较粗
2. 跨语言检索不稳（中文 query 对英文论文）
3. 拒答依赖 LLM 自检，延迟增加
4. CPU 推理较慢
5. 无 Docker 部署

## 八、课程对应

- 模块一：文档处理与检索层 → retriever/
- 模块二：RAG 生成层 → rag/
- 模块三：Agent 决策层 → agent/ + tools/
- 模块四：系统集成与前端 → ui/
- 模块五：评测与交付 → experiments/

## 九、致谢

南京农业大学课程实践项目。