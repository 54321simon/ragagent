"""全系统统一数据结构。任何模块的输入输出都必须是这里的类型。"""
from dataclasses import dataclass, field
from typing import Any, Optional, List


@dataclass
class RetrievedChunk:
    """检索返回的单个文本块。引用溯源依赖 doc_name + page。"""
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    score: float = 0.0

    def cite(self) -> str:
        """生成引用标记，如【paper.pdf-第3页】"""
        return f"【{self.doc_name}-第{self.page}页】"


@dataclass
class ToolResult:
    """工具调用的统一返回值。Agent 的 Observation 用它。"""
    tool_name: str
    success: bool
    data: Any
    elapsed_ms: float
    error: Optional[str] = None

    def to_observation(self, max_len: int = 500) -> str:
        if not self.success:
            return f"[工具错误] {self.error}"
        text = str(self.data)
        return text[:max_len] + ("..." if len(text) > max_len else "")


@dataclass
class AgentStep:
    """Agent 单步推理轨迹。前端可视化依赖它。"""
    step_idx: int
    thought: str
    action: str
    action_input: dict
    observation: str
    elapsed_ms: float = 0.0


@dataclass
class ChatResponse:
    """一次完整问答的返回。"""
    answer: str
    citations: List[RetrievedChunk] = field(default_factory=list)
    agent_trace: List[AgentStep] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)