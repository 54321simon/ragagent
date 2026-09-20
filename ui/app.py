"""智能科研助理 —— Streamlit 主应用。"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from ui.styles import inject_css
from ui.components import (
    render_trace,
    render_citations,
    render_metrics,
    render_doc_list,
)
from retriever.api import ingest_pdf, get_docs, remove_doc, get_retrieval_mode
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


# ============ 额外美化 CSS ============
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

div[data-testid="stChatMessage"] {
    background: #fafbfc;
    border: 1px solid #eaecef;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.citation {
    display: inline-block;
    background: #eef4ff;
    color: #1a56db;
    border: 1px solid #d0e0ff;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.85em;
    margin: 3px 6px 3px 0;
    font-weight: 500;
}
.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
}
.metric-value { font-size: 1.6em; font-weight: 700; color: #1a56db; line-height: 1.2; }
.metric-label { font-size: 0.8em; color: #64748b; margin-top: 4px; font-weight: 500; }

.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 3px 10px rgba(0,0,0,0.08); }

section[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px;
    border: 2px dashed #cbd5e1;
    background: #fafbfc;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1a56db;
    background: #f0f6ff;
}

.doc-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 12px; border-radius: 8px;
    background: #fafbfc; border: 1px solid #eaecef; margin-bottom: 6px;
}
.scope-badge {
    display: inline-block; background: #eef4ff; color: #1a56db;
    border-radius: 6px; padding: 4px 12px; font-size: 0.85em; font-weight: 500;
}
hr { margin: 1rem 0; border: none; border-top: 1px solid #eaecef; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============ Session State 初始化 ============
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "trace" not in st.session_state:
    st.session_state.trace = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "citations" not in st.session_state:
    st.session_state.citations = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{int(time.time())}"
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = None
if "ingested_keys" not in st.session_state:
    st.session_state.ingested_keys = set()
if "health_checks" not in st.session_state:
    st.session_state.health_checks = None


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("### 🔬 智能科研助理")
    st.caption(f"会话 `{st.session_state.session_id[:20]}`")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 新会话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.trace = []
            st.session_state.metrics = {}
            st.session_state.citations = []
            st.session_state.session_id = f"session-{int(time.time())}"
            st.rerun()
    with col_b:
        if st.button("🗑️ 清缓存", use_container_width=True):
            st.session_state.ingested_keys = set()
            st.toast("已清空上传缓存", icon="✅")
            time.sleep(0.8)
            st.rerun()

    # 系统检查按钮
    if st.button("🩺 系统检查", use_container_width=True):
        with st.spinner("检查中..."):
            from ui.health import run_all_checks
            st.session_state.health_checks = run_all_checks()

    if st.session_state.get("health_checks"):
        st.markdown("**检查结果**")
        for c in st.session_state.health_checks:
            icon = "✅" if c["ok"] else "❌"
            st.caption(f"{icon} **{c['name']}**：{c['message']}")

    st.divider()
    st.markdown("**📊 系统状态**")

    st.markdown(f"""
    <div style="font-size:0.85em; line-height:1.8;">
    <div>🧠 模型 <span style="float:right; color:#1a56db; font-weight:600;">qwen2.5:7b</span></div>
    <div>🎯 档位 <span style="float:right; color:#1a56db; font-weight:600;">{get_retrieval_mode()}</span></div>
    <div>🔧 工具 <span style="float:right; color:#1a56db; font-weight:600;">{len(get_all_tools())}</span></div>
    <div>📄 文档 <span style="float:right; color:#1a56db; font-weight:600;">{len(get_docs())}</span></div>
    <div>💭 记忆 <span style="float:right; color:#1a56db; font-weight:600;">{len(st.session_state.history) // 2} 轮</span></div>
    </div>
    """, unsafe_allow_html=True)


# ============ 三栏布局 ============
col_left, col_mid, col_right = st.columns([1, 2.5, 1.5], gap="medium")


# ---------- 左侧：知识库管理 ----------
with col_left:
    st.markdown("### 📚 知识库")

    uploaded = st.file_uploader(
        "上传论文",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="支持 PDF / Word / TXT / Markdown，可批量上传",
    )

    if uploaded:
        ingested_keys = st.session_state.ingested_keys
        new_files = [(f, f"{f.name}_{f.size}") for f in uploaded
                     if f"{f.name}_{f.size}" not in ingested_keys]

        if new_files:
            ingested_count = 0
            for f, key in new_files:
                save_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "papers", f.name,
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())

                with st.spinner(f"入库 {f.name}..."):
                    try:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext in (".pdf", ".docx", ".txt", ".md"):
                            n = ingest_pdf(save_path)
                            st.success(f"✅ {f.name} · {n} chunks")
                            ingested_keys.add(key)
                            ingested_count += 1
                        else:
                            st.warning(f"{f.name}: 不支持的格式 {ext}")
                            ingested_keys.add(key)
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

            if ingested_count > 0:
                time.sleep(1.2)
                st.rerun()
        else:
            st.caption("· 文件已入库")

    st.divider()
    st.markdown("### 🎯 检索范围")

    docs_for_select = get_docs()
    all_doc_ids = [d["doc_id"] for d in docs_for_select]

    selected = st.multiselect(
        "选择要提问的文档",
        options=all_doc_ids,
        default=st.session_state.selected_docs or [],
        label_visibility="collapsed",
        placeholder="不选 = 全部文档",
        help="不选 = 全部文档；选 1 篇 = 限定；选多篇 = 范围内检索",
    )
    st.session_state.selected_docs = selected if selected else None

    if st.session_state.selected_docs:
        docs_str = " · ".join(st.session_state.selected_docs)
        st.markdown(
            f'<div class="scope-badge">📌 {docs_str}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="scope-badge">📌 全部文档</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 📄 已上传")

    docs = get_docs()

    def _on_delete(doc_id):
        remove_doc(doc_id)
        st.toast(f"已删除 {doc_id}", icon="🗑️")
        time.sleep(0.8)
        st.rerun()

    render_doc_list(docs, on_delete=_on_delete)


# ---------- 中央：对话窗口 ----------
with col_mid:
    st.markdown("### 💬 对话")

    if st.session_state.selected_docs:
        scope = " · ".join(st.session_state.selected_docs)
    else:
        scope = "全部文档"
    st.caption(f"🎯 当前检索范围：**{scope}**")

    st.write("")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent 推理中..."):
                try:
                    result = react_loop(
                        question,
                        verbose=False,
                        history=st.session_state.history,
                        doc_ids=st.session_state.get("selected_docs"),
                    )
                except Exception as e:
                    st.error(f"Agent 调用失败：{e}")
                    result = {
                        "answer": f"抱歉，出错了：{e}",
                        "trace": [],
                        "metrics": {},
                        "success": False,
                        "history_append": [],
                    }

            st.markdown(result["answer"])

            citations = []
            for step in result.get("trace", []):
                if step.action in ("rag_search", "paper_compare") and step.observation:
                    for m in re.finditer(r"【([^】]+?)-第(\d+)页】", step.observation):
                        citations.append({
                            "doc_name": m.group(1),
                            "page": int(m.group(2)),
                        })

            if citations:
                st.write("")
                unique_cites = {}
                for c in citations:
                    unique_cites[(c["doc_name"], c["page"])] = c
                st.markdown("**📚 引用来源**")
                html = " ".join(
                    f'<span class="citation">【{doc}-第{page}页】</span>'
                    for (doc, page) in unique_cites
                )
                st.markdown(html, unsafe_allow_html=True)

            if result.get("metrics"):
                st.write("")
                render_metrics(result["metrics"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.trace = result.get("trace", [])
        st.session_state.metrics = result.get("metrics", {})

        st.session_state.history.extend(result.get("history_append", []))
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]


# ---------- 右侧：推理轨迹 ----------
with col_right:
    st.markdown("### 🔍 推理轨迹")
    st.caption("Thought → Action → Observation")
    st.write("")
    render_trace(st.session_state.trace)