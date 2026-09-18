"""前端自定义 CSS。"""

CUSTOM_CSS = """
<style>
    /* 主容器收紧间距 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 聊天消息 */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #4CAF50;
    }
    .chat-message.user {
        background-color: #f0f7ff;
        border-left-color: #2196F3;
    }
    .chat-message.assistant {
        background-color: #f0fff4;
        border-left-color: #4CAF50;
    }

    /* 推理轨迹卡片 */
    .trace-card {
        background: #fafafa;
        border-left: 4px solid #FF9800;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.9em;
    }
    .trace-card .trace-thought {
        color: #555;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .trace-card .trace-action {
        color: #1976D2;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .trace-card .trace-obs {
        color: #388E3C;
        font-size: 0.85em;
        background: #fff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 200px;
        overflow-y: auto;
    }

    /* 引用标签 */
    .citation {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85em;
        margin-right: 0.5rem;
        margin-bottom: 0.25rem;
    }

    /* 指标卡片 */
    .metric-card {
        background: #f5f5f5;
        padding: 0.75rem;
        border-radius: 0.25rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #1976D2;
    }
    .metric-label {
        font-size: 0.85em;
        color: #666;
    }

    /* 隐藏 Streamlit 默认页脚 */
    footer { visibility: hidden; }
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)