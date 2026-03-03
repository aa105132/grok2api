from __future__ import annotations

import uuid
from typing import Any, Dict, List

from app.core.config import get_config
from .models import PlannedToolCall, PlannedToolFunction, ToolCallPlanResult
from .parser import ToolCallPayloadError, parse_tool_call_payload


class ToolCallPlanner:
    """规则版 ToolCall Planner（前缀门控）"""

    def __init__(self, prefix: str | None = None, strict: bool | None = None):
        self.prefix = prefix or get_config("chat.tool_call_prefix", "<|tool_call|>")
        if strict is None:
            strict = bool(get_config("chat.tool_call_strict", True))
        self.strict = strict

    def plan(
        self,
        model_output: str,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> ToolCallPlanResult:
        text = model_output if isinstance(model_output, str) else ""
        content = text.strip()
        if not content.startswith(self.prefix):
            return ToolCallPlanResult(is_tool_call=False, raw_content=text)

        try:
            calls = parse_tool_call_payload(content, self.prefix)
        except ToolCallPayloadError as e:
            if self.strict:
                return ToolCallPlanResult(
                    is_tool_call=True,
                    tool_calls=[],
                    raw_content=text,
                    error=str(e),
                )
            return ToolCallPlanResult(is_tool_call=False, raw_content=text)

        if not calls:
            return ToolCallPlanResult(is_tool_call=False, raw_content=text)

        allowed_tools = {((tool or {}).get("function") or {}).get("name") for tool in (tools or [])}
        allowed_tools.discard(None)

        if isinstance(tool_choice, dict):
            forced_name = ((tool_choice.get("function") or {}).get("name") or "").strip()
        else:
            forced_name = ""

        normalized: List[PlannedToolCall] = []
        for idx, call in enumerate(calls):
            fn = call.get("function", {})
            fn_name = str(fn.get("name", "")).strip()

            if allowed_tools and fn_name not in allowed_tools:
                if self.strict:
                    return ToolCallPlanResult(
                        is_tool_call=True,
                        tool_calls=[],
                        raw_content=text,
                        error=f"tool `{fn_name}` not found in tools",
                    )
                return ToolCallPlanResult(is_tool_call=False, raw_content=text)

            if forced_name and fn_name != forced_name:
                if self.strict:
                    return ToolCallPlanResult(
                        is_tool_call=True,
                        tool_calls=[],
                        raw_content=text,
                        error=f"tool `{fn_name}` does not match required `{forced_name}`",
                    )
                return ToolCallPlanResult(is_tool_call=False, raw_content=text)

            call_id = (call.get("id") or "").strip() or f"call_{uuid.uuid4().hex[:24]}_{idx}"
            normalized.append(
                PlannedToolCall(
                    id=call_id,
                    type="function",
                    function=PlannedToolFunction(
                        name=fn_name,
                        arguments=str(fn.get("arguments", "{}")),
                    ),
                )
            )

        return ToolCallPlanResult(is_tool_call=True, tool_calls=normalized, raw_content=text)


__all__ = ["ToolCallPlanner"]
