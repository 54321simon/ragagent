"""智能科研助理 —— Streamlit 主应用。"""

import json
import html
import uuid
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
from common.config import MODEL
from retriever.api import ingest_with_retry, get_docs, remove_doc, get_retrieval_mode
from agent.react_loop import react_loop_stream, trim_history
from tools.registry import get_all_tools


# ============ 辅助：事件列表 → AgentStep 列表 ============
def _as_steps(events: list) -> list:
    from common.schemas import AgentStep

    steps = []
    idx = 0
    for ev in events:
        kind = ev.get("kind")
        if kind == "thought":
            steps.append(
                AgentStep(
                    step_idx=idx,
                    thought=ev["content"],
                    action="",
                    action_input={},
                    observation="",
                    elapsed_ms=0,
                )
            )
        elif kind == "action":
            steps.append(
                AgentStep(
                    step_idx=idx,
                    thought="",
                    action=ev["tool"],
                    action_input=ev.get("args", {}),
                    observation="",
                    elapsed_ms=0,
                )
            )
            idx += 1
        elif kind == "observation":
            if steps:
                steps[-1].observation = ev["content"][:500]
                steps[-1].elapsed_ms = ev.get("elapsed_ms", 0)
    return steps


# ============ 辅助：纯 HTML 渲染 trace ============
def _render_trace_html(events: list) -> str:
    if not events:
        return ""
    parts = []
    for original in events:
        ev = {
            k: html.escape(v) if isinstance(v, str) else v for k, v in original.items()
        }
        kind = ev.get("kind")
        if kind == "thought":
            parts.append(
                f'<div style="margin:6px 0;padding:8px 10px;'
                f"background:#f8fafc;border-left:3px solid #94a3b8;"
                f'border-radius:4px;font-size:0.88em;">'
                f"<b>💭 思考</b><br>{ev['content']}</div>"
            )
        elif kind == "action":
            args_str = html.escape(json.dumps(ev.get("args", {}), ensure_ascii=False))
            parts.append(
                f'<div style="margin:6px 0;padding:8px 10px;'
                f"background:#eff6ff;border-left:3px solid #3b82f6;"
                f'border-radius:4px;font-size:0.88em;">'
                f"<b>🎯 动作</b>：<code>{ev['tool']}</code><br>"
                f'<span style="color:#475569;">输入：{args_str}</span></div>'
            )
        elif kind == "actions":
            items_html = "".join(
                f'<div style="margin:4px 0;padding:6px 8px;'
                f'background:#eef2ff;border-radius:4px;font-size:0.85em;">'
                f"<b>🎯</b> <code>{html.escape(it['tool'])}</code> "
                f'<span style="color:#64748b;">'
                f"{html.escape(json.dumps(it.get('args', {}), ensure_ascii=False))}</span>"
                f"</div>"
                for it in ev.get("items", [])
            )
            parts.append(
                f'<div style="margin:6px 0;padding:8px 10px;'
                f"background:#eff6ff;border-left:3px solid #3b82f6;"
                f'border-radius:4px;font-size:0.88em;">'
                f"<b>⚡ 并行调用（{len(ev.get('items', []))} 个）</b>"
                f"{items_html}</div>"
            )
        elif kind == "observation":
            obs = (ev.get("content") or "")[:300]
            ms = ev.get("elapsed_ms", 0)
            parts.append(
                f'<div style="margin:6px 0;padding:8px 10px;'
                f"background:#f0fdf4;border-left:3px solid #22c55e;"
                f'border-radius:4px;font-size:0.88em;">'
                f"<b>👁 观察</b>（{ms:.0f}ms）<br>"
                f'<span style="color:#475569;">{obs}</span></div>'
            )
        elif kind == "action_result":
            obs = (ev.get("observation") or "")[:200]
            ms = ev.get("elapsed_ms", 0)
            ok = ev.get("ok", True)
            color = "#22c55e" if ok else "#ef4444"
            icon = "✓" if ok else "✗"
            parts.append(
                f'<div style="margin:4px 0;padding:6px 8px;'
                f"background:#f8fafc;border-left:3px solid {color};"
                f'border-radius:4px;font-size:0.85em;">'
                f"<b>{icon} {ev['tool']}</b> （{ms:.0f}ms）<br>"
                f'<span style="color:#475569;">{obs}</span></div>'
            )
    return "".join(parts)


# ============ 辅助：答案 HTML（引用替换） ============
def _render_answer_html(text: str) -> str:
    if not text:
        return ""

    def repl(m):
        doc, page = m.group(1), m.group(2)
        return f'<span class="citation">【{doc}-第{page}页】</span>'

    return re.sub(r"【([^】]+?)-第(\d+)页】", repl, html.escape(text))


# ============ 辅助：指标 HTML ============
def render_metrics_html(metrics: dict) -> str:
    items = [
        ("推理轮次", metrics.get("iterations", "-")),
        ("总代币", metrics.get("total_tokens", "-")),
        ("总耗时", f"{metrics.get('elapsed_ms', 0) / 1000:.1f}s"),
    ]
    cells = "".join(
        f'<div style="flex:1;background:linear-gradient(135deg,#f8fafc,#eef2f7);'
        f"border:1px solid #e2e8f0;border-radius:10px;padding:14px 10px;"
        f'text-align:center;margin:0 4px;">'
        f'<div style="font-size:1.5em;font-weight:700;color:#1a56db;">{v}</div>'
        f'<div style="font-size:0.8em;color:#64748b;margin-top:4px;">{k}</div>'
        f"</div>"
        for k, v in items
    )
    return f'<div style="display:flex;gap:8px;">{cells}</div>'


# ============ 辅助：工具统计面板 ============
def render_tool_stats_html(tool_stats: dict) -> str:
    if not tool_stats:
        return '<div style="color:#94a3b8;font-size:0.9em;">暂无统计</div>'

    meta = tool_stats.get("_meta", {})
    tools = {k: v for k, v in tool_stats.items() if k != "_meta"}

    rows = []
    for name, s in sorted(tools.items(), key=lambda kv: kv[1]["calls"], reverse=True):
        calls = s["calls"]
        ok = s["success"]
        rate = (ok / calls * 100) if calls else 0.0
        avg_ms = (s["total_ms"] / calls) if calls else 0.0
        rows.append(
            f"<tr>"
            f'<td style="padding:6px 8px;border-bottom:1px solid #eaecef;">'
            f"<code>{name}</code></td>"
            f'<td style="padding:6px 8px;border-bottom:1px solid #eaecef;'
            f'text-align:center;">{calls}</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eaecef;'
            f'text-align:center;color:{"#16a34a" if rate >= 80 else "#dc2626"};">'
            f"{rate:.0f}%</td>"
            f'<td style="padding:6px 8px;border-bottom:1px solid #eaecef;'
            f'text-align:right;">{avg_ms:.0f}ms</td>'
            f'<td style="padding:6px 8px;border-bottom:1px solid #eaecef;'
            f'text-align:right;">{s["tokens"]}</td>'
            f"</tr>"
        )

    if not rows:
        rows.append(
            '<tr><td colspan="5" style="padding:12px;text-align:center;'
            'color:#94a3b8;">本轮未调用任何工具</td></tr>'
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.88em;">'
        '<thead><tr style="background:#f8fafc;">'
        '<th style="padding:8px;text-align:left;border-bottom:2px solid #e2e8f0;">工具</th>'
        '<th style="padding:8px;text-align:center;border-bottom:2px solid #e2e8f0;">调用</th>'
        '<th style="padding:8px;text-align:center;border-bottom:2px solid #e2e8f0;">成功率</th>'
        '<th style="padding:8px;text-align:right;border-bottom:2px solid #e2e8f0;">平均耗时</th>'
        '<th style="padding:8px;text-align:right;border-bottom:2px solid #e2e8f0;">代币</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    rag_total = meta.get("rag_total", 0)
    rag_hit = meta.get("rag_hit", 0)
    rag_rate = (rag_hit / rag_total * 100) if rag_total else 0.0
    rag_html = (
        f'<div style="margin-top:10px;padding:10px 12px;'
        f"background:#f0f9ff;border-left:3px solid #0ea5e9;"
        f'border-radius:4px;font-size:0.9em;">'
        f"<b>🎯 RAG 命中率</b>：{rag_hit} / {rag_total} "
        f"（{rag_rate:.0f}%）</div>"
    )

    final_tokens = meta.get("final_tokens", 0)
    final_html = (
        f'<div style="margin-top:6px;padding:10px 12px;'
        f"background:#fefce8;border-left:3px solid #eab308;"
        f'border-radius:4px;font-size:0.9em;">'
        f"<b>✍️ 最终答案 Token</b>：{final_tokens}</div>"
    )

    return table + rag_html + final_html


# ============ 会话管理 ============
def _new_session(title: str = "新会话") -> dict:
    return {
        "id": f"session-{uuid.uuid4().hex}",
        "title": title,
        "created_at": time.time(),
        "messages": [],
        "history": [],
        "history_summary": "",
        "trace": [],
        "metrics": {},
        "tool_stats": {},
        "selected_docs": None,
    }


def _ensure_session_state():
    if "sessions" not in st.session_state:
        s = _new_session("新会话")
        st.session_state.sessions = {s["id"]: s}
        st.session_state.current_session_id = s["id"]
    if "current_session_id" not in st.session_state:
        first_id = next(iter(st.session_state.sessions))
        st.session_state.current_session_id = first_id
    if st.session_state.current_session_id not in st.session_state.sessions:
        if st.session_state.sessions:
            st.session_state.current_session_id = next(iter(st.session_state.sessions))
        else:
            s = _new_session("新会话")
            st.session_state.sessions[s["id"]] = s
            st.session_state.current_session_id = s["id"]

    if "ingested_keys" not in st.session_state:
        st.session_state.ingested_keys = set()
    if "health_checks" not in st.session_state:
        st.session_state.health_checks = None


def _cur_session() -> dict:
    return st.session_state.sessions[st.session_state.current_session_id]


# ============ 页面配置 ============
st.set_page_config(
    page_title="智能科研助理",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
_ensure_session_state()


# ============ 额外美化 CSS ============
st.markdown(
    """
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
.session-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 10px; border-radius: 6px;
    background: #fafbfc; border: 1px solid #eaecef; margin-bottom: 4px;
    font-size: 0.88em;
}
.session-item.active {
    background: #eef4ff; border-color: #d0e0ff; color: #1a56db; font-weight: 600;
}
hr { margin: 1rem 0; border: none; border-top: 1px solid #eaecef; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("### 🔬 智能科研助理")

    if st.button("➕ 新建会话", use_container_width=True, type="primary"):
        s = _new_session("新会话")
        st.session_state.sessions[s["id"]] = s
        st.session_state.current_session_id = s["id"]
        st.rerun()

    st.divider()
    st.markdown("**💬 会话历史**")

    session_items = sorted(
        st.session_state.sessions.values(),
        key=lambda x: x["created_at"],
        reverse=True,
    )

    for s in session_items:
        sid = s["id"]
        is_active = sid == st.session_state.current_session_id
        title = s.get("title") or "新会话"
        title_show = title[:14] + ("…" if len(title) > 14 else "")

        col_s, col_d = st.columns([5, 1])
        with col_s:
            label = f"{'● ' if is_active else ''}{title_show}"
            if st.button(label, key=f"switch_{sid}", use_container_width=True):
                if not is_active:
                    st.session_state.current_session_id = sid
                    st.rerun()
        with col_d:
            if st.button("🗑", key=f"del_{sid}", use_container_width=True):
                st.session_state.sessions.pop(sid, None)
                if sid == st.session_state.current_session_id:
                    if st.session_state.sessions:
                        latest = max(
                            st.session_state.sessions.values(),
                            key=lambda x: x["created_at"],
                        )
                        st.session_state.current_session_id = latest["id"]
                    else:
                        s2 = _new_session("新会话")
                        st.session_state.sessions[s2["id"]] = s2
                        st.session_state.current_session_id = s2["id"]
                st.rerun()

    st.divider()

    if st.button("🗑️ 清缓存", use_container_width=True):
        st.session_state.ingested_keys = set()
        from rag.cache import SemanticCache

        SemanticCache().clear()
        st.toast("已清空上传缓存", icon="✅")
        time.sleep(0.8)
        st.rerun()

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

    cur = _cur_session()
    st.markdown(
        f"""
    <div style="font-size:0.85em; line-height:1.8;">
    <div>🧠 模型 <span style="float:right; color:#1a56db; font-weight:600;">{html.escape(MODEL)}</span></div>
    <div>🎯 档位 <span style="float:right; color:#1a56db; font-weight:600;">{get_retrieval_mode()}</span></div>
    <div>🔧 工具 <span style="float:right; color:#1a56db; font-weight:600;">{len(get_all_tools())}</span></div>
    <div>📄 文档 <span style="float:right; color:#1a56db; font-weight:600;">{len(get_docs())}</span></div>
    <div>💭 记忆 <span style="float:right; color:#1a56db; font-weight:600;">{len(cur["history"]) // 2} 轮</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


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
        new_files = [
            (f, f"{f.name}_{f.size}")
            for f in uploaded
            if f"{f.name}_{f.size}" not in ingested_keys
        ]

        if new_files:
            ingested_count = 0
            progress = st.progress(0, text="准备批量入库")
            for file_index, (f, key) in enumerate(new_files):
                if f.size > 32 * 1024 * 1024:
                    st.error(f"{f.name} 超过 32 MB，请拆分后上传。")
                    continue
                save_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data",
                    "papers",
                    os.path.basename(f.name),
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as out:
                    out.write(f.getbuffer())

                with st.spinner(f"入库 {f.name}..."):
                    try:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext in (".pdf", ".docx", ".txt", ".md"):
                            n = ingest_with_retry(save_path)
                            st.success(f"✅ {f.name} · {n} chunks")
                            ingested_keys.add(key)
                            ingested_count += 1
                        else:
                            st.warning(f"{f.name}: 不支持的格式 {ext}")
                            ingested_keys.add(key)
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

                progress.progress(
                    (file_index + 1) / len(new_files),
                    text=f"已处理 {file_index + 1}/{len(new_files)}",
                )

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
        default=[
            d for d in (_cur_session().get("selected_docs") or []) if d in all_doc_ids
        ],
        label_visibility="collapsed",
        placeholder="不选 = 全部文档",
        help="不选 = 全部文档；选 1 篇 = 限定；选多篇 = 范围内检索",
    )
    _cur_session()["selected_docs"] = selected if selected else None

    if _cur_session()["selected_docs"]:
        docs_str = " · ".join(_cur_session()["selected_docs"])
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

    cur = _cur_session()

    if cur.get("selected_docs"):
        scope = " · ".join(cur["selected_docs"])
    else:
        scope = "全部文档"
    st.caption(f"🎯 当前检索范围：**{scope}**")

    st.write("")

    for msg in cur["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("输入你的问题..."):
        if cur["title"] == "新会话" or not cur["title"]:
            cur["title"] = question[:30]

        cur["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_ph = st.empty()
            cite_ph = st.empty()
            metric_ph = st.empty()

            live_trace = []
            full_text = ""
            done_payload = None

            FLUSH_INTERVAL_MS = 60
            last_answer_flush = 0.0

            trace_ph = st.session_state.get("_trace_ph")

            try:
                for event in react_loop_stream(
                    question,
                    history=cur["history"],
                    history_summary=cur.get("history_summary", ""),
                    doc_ids=cur.get("selected_docs"),
                    session_id=st.session_state.current_session_id,
                ):
                    etype = event.get("type")
                    now_ms = time.time() * 1000

                    if etype == "thought":
                        live_trace.append(
                            {
                                "kind": "thought",
                                "content": event["content"],
                            }
                        )
                        if trace_ph is not None:
                            trace_ph.markdown(
                                _render_trace_html(live_trace),
                                unsafe_allow_html=True,
                            )

                    elif etype == "action":
                        live_trace.append(
                            {
                                "kind": "action",
                                "tool": event["tool"],
                                "args": event["args"],
                            }
                        )
                        if trace_ph is not None:
                            trace_ph.markdown(
                                _render_trace_html(live_trace),
                                unsafe_allow_html=True,
                            )

                    elif etype == "actions":
                        live_trace.append(
                            {
                                "kind": "actions",
                                "items": event["items"],
                            }
                        )
                        if trace_ph is not None:
                            trace_ph.markdown(
                                _render_trace_html(live_trace),
                                unsafe_allow_html=True,
                            )

                    elif etype == "observation":
                        live_trace.append(
                            {
                                "kind": "observation",
                                "content": event["content"],
                                "elapsed_ms": event.get("elapsed_ms", 0),
                            }
                        )
                        if trace_ph is not None:
                            trace_ph.markdown(
                                _render_trace_html(live_trace),
                                unsafe_allow_html=True,
                            )

                    elif etype == "action_result":
                        live_trace.append(
                            {
                                "kind": "action_result",
                                "tool": event["tool"],
                                "observation": event["observation"],
                                "elapsed_ms": event.get("elapsed_ms", 0),
                                "ok": event.get("ok", True),
                            }
                        )
                        if trace_ph is not None:
                            trace_ph.markdown(
                                _render_trace_html(live_trace),
                                unsafe_allow_html=True,
                            )

                    elif etype == "observations":
                        # 全部并行完成，无需额外操作（前面已逐个刷过）
                        pass

                    elif etype == "final_chunk":
                        full_text += event["content"]
                        if now_ms - last_answer_flush >= FLUSH_INTERVAL_MS:
                            answer_ph.markdown(
                                full_text + '<span style="color:#1a56db;">▌</span>',
                                unsafe_allow_html=False,
                            )
                            last_answer_flush = now_ms

                    elif etype == "done":
                        done_payload = event

            except Exception as e:
                st.error(f"Agent 调用失败：{e}")
                done_payload = {
                    "answer": f"抱歉，出错了：{e}",
                    "trace": [],
                    "metrics": {},
                    "tool_stats": {},
                    "success": False,
                    "history_append": [],
                }

            answer_ph.markdown(
                full_text,
                unsafe_allow_html=False,
            )

            if done_payload:
                citations = []
                for step in done_payload.get("trace", []):
                    act = getattr(step, "action", None)
                    obs = getattr(step, "observation", None)
                    if act in ("rag_search", "paper_compare") and obs:
                        for m in re.finditer(r"【([^】]+?)-第(\d+)页】", obs):
                            citations.append(
                                {
                                    "doc_name": m.group(1),
                                    "page": int(m.group(2)),
                                }
                            )

                if citations:
                    unique_cites = {}
                    for c in citations:
                        unique_cites[(c["doc_name"], c["page"])] = c
                    cite_ph.markdown("**📚 引用来源**")
                    html_cites = " ".join(
                        f'<span class="citation">【{html.escape(doc)}-第{page}页】</span>'
                        for (doc, page) in unique_cites
                    )
                    cite_ph.markdown(html_cites, unsafe_allow_html=True)

                if done_payload.get("metrics"):
                    metric_ph.markdown(
                        render_metrics_html(done_payload["metrics"]),
                        unsafe_allow_html=True,
                    )

        # ===== 写回当前会话 =====
        final_answer = done_payload.get("answer", "") if done_payload else ""
        cur["messages"].append({"role": "assistant", "content": final_answer})
        cur["trace"] = done_payload.get("trace", []) if done_payload else []
        cur["metrics"] = done_payload.get("metrics", {}) if done_payload else {}
        cur["tool_stats"] = done_payload.get("tool_stats", {}) if done_payload else {}

        cur["history"].extend(
            done_payload.get("history_append", []) if done_payload else []
        )
        try:
            new_hist, new_summary = trim_history(
                cur["history"],
                existing_summary=cur.get("history_summary", ""),
            )
            cur["history"] = new_hist
            cur["history_summary"] = new_summary
        except Exception:
            if len(cur["history"]) > 20:
                cur["history"] = cur["history"][-20:]

        if cur["title"] == question[:30] or cur["title"] == "新会话":
            cur["title"] = question[:30]

        st.session_state["_trace_ph"] = None
        st.rerun()


# ---------- 右侧：推理轨迹 ----------
with col_right:
    st.markdown("### 🔍 推理轨迹")
    st.caption("Thought → Action → Observation")
    st.write("")
    st.session_state["_trace_ph"] = st.empty()
    render_trace(_cur_session().get("trace", []))

    st.divider()
    st.markdown("### 📊 指标统计")
    st.caption("RAG 命中表示检索返回证据，真实检索准确率见评测报告。")
    if _cur_session().get("metrics", {}).get("missing_citations"):
        st.warning("本轮模型未逐句标注引用，请核对答案末尾的检索来源。")
    st.caption("按工具聚合 · 当前会话最近一轮")
    stats = _cur_session().get("tool_stats") or {}
    st.markdown(render_tool_stats_html(stats), unsafe_allow_html=True)
