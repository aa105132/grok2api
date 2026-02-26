from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlannedToolFunction:
    """工具函数调用信息"""

    name: str
    arguments: str


@dataclass
class PlannedToolCall:
    """规划后的工具调用"""

    id: str
    type: str = "function"
    function: PlannedToolFunction = field(default_factory=lambda: PlannedToolFunction(name="", arguments="{}"))

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class ToolCallPlanResult:
    """工具调用规划结果"""

    is_tool_call: bool
    tool_calls: List[PlannedToolCall] = field(default_factory=list)
    raw_content: str = ""
    error: Optional[str] = None


__all__ = [
    "PlannedToolFunction",
    "PlannedToolCall",
    "ToolCallPlanResult",
]
