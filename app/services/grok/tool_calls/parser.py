from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import orjson


class ToolCallPayloadError(ValueError):
    """工具调用负载解析失败"""


def _strip_prefix(raw_text: str, prefix: str) -> Optional[str]:
    if not isinstance(raw_text, str):
        return None
    content = raw_text.strip()
    if not content.startswith(prefix):
        return None
    return content[len(prefix) :].strip()


def _normalize_single_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    function = payload.get("function")
    if not isinstance(function, dict):
        raise ToolCallPayloadError("missing function object")

    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolCallPayloadError("missing function.name")

    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = arguments.strip() or "{}"
        try:
            orjson.loads(arguments)
        except Exception as e:  # noqa: BLE001
            raise ToolCallPayloadError(f"invalid function.arguments json string: {e}") from e
        normalized_arguments = arguments
    elif isinstance(arguments, dict):
        normalized_arguments = orjson.dumps(arguments).decode("utf-8")
    else:
        raise ToolCallPayloadError("function.arguments must be string or object")

    call_id = payload.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = ""

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": normalized_arguments,
        },
    }


def parse_tool_call_payload(raw_text: str, prefix: str) -> List[Dict[str, Any]]:
    """
    解析带前缀的工具调用文本，返回标准化调用列表。

    支持两种负载形式：
    1) {"tool_calls": [...]} 或 {"function": {...}}
    2) [...]（tool_calls 数组）
    """

    body = _strip_prefix(raw_text, prefix)
    if body is None:
        return []

    try:
        payload = json.loads(body)
    except Exception as e:  # noqa: BLE001
        raise ToolCallPayloadError(f"invalid tool call payload json: {e}") from e

    calls: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_calls"), list):
            for item in payload["tool_calls"]:
                if not isinstance(item, dict):
                    raise ToolCallPayloadError("tool_calls item must be object")
                calls.append(_normalize_single_call(item))
        elif isinstance(payload.get("function"), dict):
            calls.append(_normalize_single_call(payload))
        else:
            raise ToolCallPayloadError("payload must contain tool_calls or function")
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ToolCallPayloadError("tool_calls item must be object")
            calls.append(_normalize_single_call(item))
    else:
        raise ToolCallPayloadError("payload must be object or array")

    return calls


__all__ = ["ToolCallPayloadError", "parse_tool_call_payload"]
