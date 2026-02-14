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


def test_stream_processor_wraps_reasoning_tokens_with_think_tags():
    responses = [
        {"responseId": "resp-1", "isThinking": True, "token": "Thinking about the user's request"},
        {"isThinking": True, "token": "问候处理- 你好作为初始问候"},
        {"isThinking": False, "token": "你好！有什么可以帮你的吗？"},
        {"modelResponse": {"responseId": "resp-1", "message": "你好！有什么可以帮你的吗？"}},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=True))

    assert "<think>\n" in text
    assert "Thinking about the user's request" in text
    assert "问候处理- 你好作为初始问候" in text
    assert "</think>\n" in text
    assert text.endswith("你好！有什么可以帮你的吗？")


def test_stream_processor_hides_reasoning_tokens_when_disabled():
    responses = [
        {"responseId": "resp-2", "isThinking": True, "token": "Thinking about the user's request"},
        {"isThinking": True, "token": "问候处理- 你好作为初始问候"},
        {"modelResponse": {"responseId": "resp-2", "message": "你好！有什么可以帮你的吗？"}},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=False))

    assert "Thinking about the user's request" not in text
    assert "问候处理- 你好作为初始问候" not in text
    assert text == "你好！有什么可以帮你的吗？"


def test_stream_processor_keeps_reasoning_wrapped_when_intermediate_flag_missing():
    responses = [
        {"responseId": "resp-3", "isThinking": True, "token": "推理第一段"},
        {"token": "推理第二段"},
        {"isThinking": False, "token": "最终答案"},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=True))

    think_start = text.index("<think>")
    think_end = text.index("</think>")
    think_block = text[think_start:think_end]

    assert "推理第一段" in think_block
    assert "推理第二段" in think_block
    assert text.endswith("最终答案")


def test_stream_processor_supports_enabled_disabled_reasoning_flags():
    responses = [
        {"responseId": "resp-4", "reasoningStatus": "enabled", "token": "推理内容"},
        {"reasoningStatus": "disabled", "token": "直接答案"},
    ]

    text = asyncio.run(_run_stream_processor(responses, think=True))

    assert "<think>" in text
    assert "推理内容" in text
    assert "</think>" in text
    assert text.endswith("直接答案")
