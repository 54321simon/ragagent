"""可复用 UI 组件。"""
import streamlit as st
from typing import List
from common.schemas import AgentStep, RetrievedChunk


def render_trace(trace: List[AgentStep]):
    """渲染 Agent 推理轨迹。"""
    if not trace:
        st.caption("提问后显示 Agent 推理过程")
        return

    for step in trace:
        action_label = step.action
        if step.action == "Final":
            action_label = "✅ Final Answer"
        elif step.action == "PARSE_ERROR":
            action_label = "⚠️ Parse Error"

        with st.expander(f"Step {step.step_idx}: {action_label}", expanded=True):
            # Thought
            st.markdown(f"**💭 Thought:** {step.thought}")

            # Action
            if step.action not in ("Final", "PARSE_ERROR"):
                st.markdown(f"**🎯 Action:** `{step.action}`")
                if step.action_input:
                    st.markdown(f"**📥 Input:** `{step.action_input}`")

            # Observation
            if step.observation:
                st.markdown("**👁️ Observation:**")
                st.code(step.observation[:500], language=None)


def render_citations(citations: List[RetrievedChunk]):
    """渲染引用列表。"""
    if not citations:
        return

    st.markdown("**📚 引用来源：**")
    seen = set()
    for c in citations:
        key = (c.doc_name, c.page)
        if key in seen:
            continue
        seen.add(key)
        st.markdown(
            f'<span class="citation">{c.cite()} · 相似度 {c.score:.3f}</span>',
            unsafe_allow_html=True,
        )


def render_metrics(metrics: dict):
    """渲染指标卡片。"""
    if not metrics:
        return

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("iterations", 0)}</div>'
            f'<div class="metric-label">推理轮次</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("total_tokens", 0)}</div>'
            f'<div class="metric-label">Tokens</div></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        elapsed = metrics.get("elapsed_ms", 0) / 1000
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{elapsed:.1f}s</div>'
            f'<div class="metric-label">总耗时</div></div>',
            unsafe_allow_html=True,
        )


def render_doc_list(docs: List[dict], on_delete=None):
    """渲染文档列表。"""
    if not docs:
        st.caption("暂无文档")
        return

    for d in docs:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"📄 **{d['doc_name']}**")
            st.caption(f"doc_id: `{d['doc_id']}`")
        with col2:
            if on_delete and st.button("🗑️", key=f"del_{d['doc_id']}"):
                on_delete(d["doc_id"])