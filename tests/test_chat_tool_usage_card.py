import asyncio

import orjson

from app.core.config import config
from app.services.grok.processors.chat_processors import StreamProcessor


def _build_response_line(payload: dict) -> bytes:
    return orjson.dumps({"result": {"response": payload}})


def _extract_text_from_sse(chunks: list[str]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        body = chunk[6:].strip()
        if not body or body == "[DONE]":
            continue
        data = orjson.loads(body)
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


async def _run_stream_processor(responses: list[dict], think: bool) -> str:
    await config.load()

    async def _response_iter():
        for item in responses:
            yield _build_response_line(item)

    processor = StreamProcessor("grok-4-1-thinking-1129", "token", think=think)
    chunks: list[str] = []
    async for chunk in processor.process(_response_iter()):
        chunks.append(chunk)

    return _extract_text_from_sse(chunks)


def test_stream_processor_ignores_chatroom_send_tool_card():
    tool_card = (
        "<xai:tool_usage_card>"
        "<xai:tool_name><![CDATA[chatroom_send]]></xai:tool_name>"
        "<xai:tool_args><![CDATA[{\"message\":\"internal think\"}]]></xai:tool_args>"
        "</xai:tool_usage_card>"
    )
    responses = [
        {"responseId": "resp-tool-1", "isThinking": False, "token": tool_card},
        {"isThinking": False, "token": "最终答案"},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=False))

    assert "AgentThink" not in text
    assert "internal think" not in text
    assert text.endswith("最终答案")


def test_stream_processor_keeps_supported_tool_cards_visible():
    tool_card = (
        "<xai:tool_usage_card>"
        "<xai:tool_name><![CDATA[web_search]]></xai:tool_name>"
        "<xai:tool_args><![CDATA[{\"query\":\"quantum computing news\"}]]></xai:tool_args>"
        "</xai:tool_usage_card>"
    )
    responses = [
        {"responseId": "resp-tool-2", "isThinking": False, "token": tool_card},
        {"isThinking": False, "token": "结果摘要"},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=False))

    assert "[WebSearch] quantum computing news" in text
    assert text.endswith("结果摘要")
