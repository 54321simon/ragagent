"""智能科研助理 —— Streamlit 主应用。"""
import os
import sys
import time

# 确保能 import 项目根目录的包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from ui.styles import inject_css
from ui.components import (
    render_trace,
    render_citations,
    render_metrics,
    render_doc_list,
)
from retriever.api import ingest_pdf, get_docs, remove_doc
from agent.react_loop import react_loop
from tools.registry import get_all_tools

# ============ 页面配置 ============
st.set_page_config(
    page_title="智能科研助理",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ============ Session State 初始化 ============
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace" not in st.session_state:
    st.session_state.trace = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "citations" not in st.session_state:
    st.session_state.citations = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{int(time.time())}"


# ============ 侧边栏：会话信息 ============
with st.sidebar:
    st.title("🔬 智能科研助理")
    st.caption(f"会话 ID: `{st.session_state.session_id}`")
    st.divider()

    if st.button("🔄 新建会话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trace = []
        st.session_state.metrics = {}
        st.session_state.citations = []
        st.session_state.session_id = f"session-{int(time.time())}"
        st.rerun()

    st.divider()
    st.markdown("**📊 系统状态**")
    st.caption(f"模型: qwen2.5:7b")
    st.caption(f"工具数: {len(get_all_tools())}")
    st.caption(f"文档数: {len(get_docs())}")


# ============ 三栏布局 ============
col_left, col_mid, col_right = st.columns([1, 2.5, 1.5])


# ---------- 左侧：知识库管理 ----------
with col_left:
    st.header("📚 知识库")

    uploaded = st.file_uploader(
        "上传论文",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="支持 PDF / Word / TXT / Markdown，可批量上传",
    )

    if uploaded:
        for f in uploaded:
            save_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "papers", f.name,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

            with st.spinner(f"正在入库 {f.name}..."):
                try:
                    if f.name.lower().endswith(".pdf"):
                        n = ingest_pdf(save_path)
                    else:
                        st.warning(f"{f.name}: 目前只支持 PDF 入库，其他格式待扩展")
                        continue
                    st.success(f"✅ {f.name} 入库 {n} 个 chunk")
                except Exception as e:
                    st.error(f"❌ {f.name} 入库失败：{e}")

    st.divider()
    st.subheader("已上传文档")

    docs = get_docs()

    def _on_delete(doc_id):
        remove_doc(doc_id)
        st.success(f"已删除 {doc_id}")
        st.rerun()

    render_doc_list(docs, on_delete=_on_delete)


# ---------- 中央：对话窗口 ----------
with col_mid:
    st.header("💬 对话")

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if question := st.chat_input("输入你的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 调用 Agent
        with st.chat_message("assistant"):
            with st.spinner("Agent 推理中..."):
                try:
                    result = react_loop(question, verbose=False)
                except Exception as e:
                    st.error(f"Agent 调用失败：{e}")
                    result = {
                        "answer": f"抱歉，出错了：{e}",
                        "trace": [],
                        "metrics": {},
                        "success": False,
                    }

            # 显示答案
            st.markdown(result["answer"])

            # 显示引用（如果有）
            citations = []
            for step in result.get("trace", []):
                # 从 rag_search 的 observation 里提取引用
                if step.action == "rag_search" and step.observation:
                    import re
                    for m in re.finditer(r"【([^】]+?)-第(\d+)页】", step.observation):
                        citations.append({
                            "doc_name": m.group(1),
                            "page": int(m.group(2)),
                        })
            if citations:
                st.markdown("---")
                unique_cites = {}
                for c in citations:
                    key = (c["doc_name"], c["page"])
                    unique_cites[key] = c
                st.markdown("**📚 引用来源：**")
                for (doc, page) in unique_cites:
                    st.markdown(
                        f'<span class="citation">【{doc}-第{page}页】</span>',
                        unsafe_allow_html=True,
                    )

            # 显示指标
            if result.get("metrics"):
                st.markdown("---")
                render_metrics(result["metrics"])

        # 保存到历史
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.trace = result.get("trace", [])
        st.session_state.metrics = result.get("metrics", {})


# ---------- 右侧：推理轨迹 ----------
with col_right:
    st.header("🔍 推理轨迹")
    st.caption("Agent 每一步 Thought → Action → Observation")

    render_trace(st.session_state.trace)


# ============ 底部：快速测试 ============
with st.expander("🧪 快速测试用例（点击填入问题）"):
    st.markdown("""
    - 你好，请介绍一下你自己
    - 3.14 乘以 2.56 等于多少？
    - 这篇论文用了什么方法？
    - 知识库里有哪些论文？
    - 论文的核心贡献是什么？
    """)